"""Build the E0 training/evaluation arrays from OUHD-L ligature units v3.0.

PYTHONPATH=<v3>/build:<v3>:. python build_e0_data.py --v3 <ouhdl_ligatures_v3> --ouhdl <ouhdl root> \
    --gold <uhwr_labels_Ali<user>_2026-10-15.json> --out <dir> [--procs 12]

Writes e0_{train,val,test,gold}.npz (see FIELDS) and e0_build_stats.json.
  train/val : unit_ids, sample_ids, writer_ids, cond_rendered (N,64,64 uint8, black on white),
              seq (S,4 float32 Nagamatsu [x,y,pen,eos]), seq_offsets (N+1), step (N,)
              + gt_points/gt_stroke_id/gt_offsets (native ink on the 64x64 eval canvas; val only)
  test/gold : the same + cond_dataset (N,64,64 uint8) and cond_*_f4 (N,256,256 uint8) for the
              geometric baseline (rendered_unit / dataset_unit at f = 4, exactly as run_ouhdl.py).
gold = <author>'s 21 hand-corrected test lines (labels 'gold' entries with done=True); unit k of a line is
the set of pen-down strokes with assign == k, in recording order; unit_id <sample_id>_G<k:03d>.
"""
import argparse
import hashlib
import json
import os
import time
from multiprocessing import get_context

import cv2
import numpy as np
import pandas as pd

import ouhdl_ligatures as OL
from uhwr_baselines import inputs as I
from uhwr_baselines import ouhdl as O
from uhwr_eval import apply_transform, fit_transform_unit

import e0_adapter as A


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def encode_unit(ink):
    arr, _, step = A.encode(ink)
    return arr, step


# ---------------------------------------------------------------- train / val (no line context)
def split_arrays(v3, split, keep_gt):
    idx, U = OL.load_split(v3, split)
    conds, seqs, steps, inks = [], [], [], []
    for i in range(len(U)):
        ink, _ = A.unit_eval_ink([s[:, :2] for s in U.get(i)])
        arr, step = encode_unit(ink)
        conds.append(A.render_condition(ink)); seqs.append(arr); steps.append(step)
        if keep_gt:
            inks.append(ink)
    out = dict(unit_ids=idx.unit_id.to_numpy().astype(str), sample_ids=idx.sample_id.to_numpy().astype(str),
               writer_ids=idx.writer_id.to_numpy(), cond_rendered=np.stack(conds),
               seq=np.vstack(seqs).astype(np.float32),
               seq_offsets=np.r_[0, np.cumsum([len(s) for s in seqs])].astype(np.int64),
               step=np.array(steps, np.float32))
    if keep_gt:
        P, S, off = A.pack_inks(inks)
        out.update(gt_points=P, gt_stroke_id=S, gt_offsets=off)
    return out


# ---------------------------------------------------------------- test / gold (line context)
def line_work(item):
    sid, csv_path, img_path, groups = item
    raw = O.load_raw_strokes(csv_path)
    L = O.make_line(sid, raw)
    img_ds = I.dataset_image('', img_path)
    res = []
    for uid, sidx in groups:
        assert sorted(sidx) == list(sidx)
        st_line = [L.ink[i] for i in sidx]
        T = fit_transform_unit(st_line, 64, 4)
        ink = apply_transform(st_line, *T)
        ink_raw, _ = A.unit_eval_ink([raw[i] for i in sidx])   # same ink computed from raw coords
        dev = max(float(np.abs(a - b).max()) for a, b in zip(ink, ink_raw))
        u = dict(unit_id=uid, stroke_idx=list(sidx), T=T, ink=ink, canvas=(64, 64))
        r4, _ = I.rendered_unit(u, 4.0)
        d4, _ = I.dataset_unit(L, u, img_ds, 4.0)
        d1 = cv2.resize(d4, (64, 64), interpolation=cv2.INTER_AREA)
        arr, step = encode_unit(ink)
        res.append(dict(unit_id=uid, sample_id=sid, ink=ink, cond_rendered=A.render_condition(ink),
                        cond_dataset=d1, cond_rendered_f4=r4, cond_dataset_f4=d4, seq=arr, step=step,
                        raw_vs_line_maxdev=dev))
    return res


def eval_arrays(items, writer_of, procs):
    with get_context('spawn').Pool(procs) as pool:
        res = [r for rr in pool.map(line_work, items, chunksize=4) for r in rr]
    P, S, off = A.pack_inks([r['ink'] for r in res])
    seqs = [r['seq'] for r in res]
    return dict(unit_ids=np.array([r['unit_id'] for r in res]), sample_ids=np.array([r['sample_id'] for r in res]),
                writer_ids=np.array([writer_of[r['sample_id']] for r in res]),
                cond_rendered=np.stack([r['cond_rendered'] for r in res]),
                cond_dataset=np.stack([r['cond_dataset'] for r in res]),
                cond_rendered_f4=np.stack([r['cond_rendered_f4'] for r in res]),
                cond_dataset_f4=np.stack([r['cond_dataset_f4'] for r in res]),
                seq=np.vstack(seqs).astype(np.float32),
                seq_offsets=np.r_[0, np.cumsum([len(s) for s in seqs])].astype(np.int64),
                step=np.array([r['step'] for r in res], np.float32),
                gt_points=P, gt_stroke_id=S, gt_offsets=off), max(r['raw_vs_line_maxdev'] for r in res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v3', required=True); ap.add_argument('--ouhdl', required=True)
    ap.add_argument('--gold', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--procs', type=int, default=12)
    ap.add_argument('--splits', default='train,val,test,gold')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time(); stats = {}
    man = O.load_manifest(a.ouhdl, 'test').set_index('sample_id')
    for split in a.splits.split(','):
        t1 = time.time()
        if split in ('train', 'val'):
            d = split_arrays(a.v3, split, keep_gt=(split == 'val'))
            dev = None
        else:
            idx, U = OL.load_split(a.v3, 'test')
            if split == 'test':
                groups = {sid: [(r.unit_id, json.loads(r.stroke_idx)) for r in g.itertuples()]
                          for sid, g in idx.groupby('sample_id', sort=False)}
            else:
                G = json.load(open(a.gold))['gold']
                groups = {}
                for sid, v in G.items():
                    if not v.get('done'):
                        continue
                    asg = v['assign']
                    groups[sid] = [(f'{sid}_G{k:03d}', [i for i, x in enumerate(asg) if x == k])
                                   for k in sorted(set(asg))]
            writer_of = idx.groupby('sample_id').writer_id.first().to_dict()
            items = [(sid, os.path.join(a.ouhdl, man.loc[sid, 'csv']), os.path.join(a.ouhdl, man.loc[sid, 'img']), g)
                     for sid, g in groups.items()]
            if split == 'gold':  # check stroke counts of labels against raw strokes
                for sid, csvp, _, g in items:
                    n = len(O.load_raw_strokes(csvp))
                    assert sum(len(s) for _, s in g) == n, (sid, n)
            d, dev = eval_arrays(items, writer_of, a.procs)
            if split == 'test':  # v3 index order and point equality with the v3 npz
                assert list(d['unit_ids']) == list(idx.unit_id), 'unit order differs from v3 index'
        f = os.path.join(a.out, f'e0_{split}.npz')
        np.savez_compressed(f, **d)
        n_seq = np.diff(d['seq_offsets'])
        stats[split] = dict(n_units=int(len(d['unit_ids'])), n_lines=int(len(set(d['sample_ids']))),
                            seq_len_pct={str(q): float(np.percentile(n_seq, q)) for q in (50, 90, 99, 100)},
                            n_step_gt_1=int((d['step'] > 1.0 + 1e-6).sum()), step_max=float(d['step'].max()),
                            raw_vs_line_max_abs_dev_px=dev, file=os.path.basename(f), sha256=sha(f),
                            seconds=round(time.time() - t1, 1))
        print(split, stats[split], flush=True)
    json.dump(dict(stats=stats, seconds=round(time.time() - t0, 1)), open(os.path.join(a.out, 'e0_build_stats.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
