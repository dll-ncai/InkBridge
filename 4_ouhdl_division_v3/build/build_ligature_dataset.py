"""Build the OUHD-L ligature dataset, division v3.0, for train / val / test.

Units come from ligseg_v3.divide_line (per-line transcript + pen-timing alignment), with the
per-letter lengths in letter_lengths.json (fitted on train only).  Every pen-down stroke of a line
belongs to exactly one unit.  Coordinates stay in raw tablet units.

Outputs in --out:
  ligatures_v3_<split>.csv        one row per unit (index; stroke indices refer to the line's
                                  pen-down strokes in recording order, as in uhwr_baselines.ouhdl)
  ligatures_v3_<split>.npz        points (N,5) int32: x, y, time, pressure, local stroke number;
                                  unit_offsets (U+1,) into points; unit_ids (U,)
  lines_v3_<split>.csv            one row per line: units, text ligatures, skipped ligatures, ...
  ouhdl_ligatures.py              loader
  PROVENANCE.json                 written by this script

Usage: PYTHONPATH=<uhwr_baselines adapter>:<uhwr_eval>:<dir of ligseg_v3.py> \
         python build_ligature_dataset.py --root <ouhdl> --lengths letter_lengths.json --out <dir>
"""
import argparse, datetime, hashlib, json, os, platform, shutil, sys, time
from multiprocessing import get_context
import numpy as np, pandas as pd

import ligseg_v3 as L3
from uhwr_baselines import ouhdl as O

COLS = ['X cood.', 'Y cood.', 'Time', 'Pressure', 'pen_down']


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def raw_strokes_ext(csv_path):
    """Same stroke segmentation as uhwr_baselines.ouhdl.load_raw_strokes, keeping time and pressure."""
    d = pd.read_csv(csv_path, usecols=COLS)
    pdn = d['pen_down'].to_numpy() == 1
    arr = d[['X cood.', 'Y cood.', 'Time', 'Pressure']].to_numpy(np.float64)
    edges = np.diff(np.r_[0, pdn.astype(int), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    strokes = [arr[s:e] for s, e in zip(starts, ends)]
    strokes = [s[np.isfinite(s[:, :2]).all(1)] for s in strokes]
    return [s for s in strokes if len(s)]


def work(item):
    sid, csv, text, LEN = item
    raw = O.load_raw_strokes(csv)
    ext = raw_strokes_ext(csv)
    assert len(raw) == len(ext) and all(np.array_equal(a, b[:, :2]) for a, b in zip(raw, ext)), f'{sid}: stroke mismatch'
    ln = O.make_line(sid, raw)
    units, info = L3.divide_line(ln.ink, text, LEN)
    cover = sorted(i for u in units for i in u['stroke_idx'])
    assert cover == list(range(len(raw))), f'{sid}: strokes not covered exactly once'
    return sid, units, info, ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--lengths', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--procs', type=int, default=12); ap.add_argument('--check-test', help='units_v3_test.json to compare against')
    a = ap.parse_args(); t0 = time.time()
    LEN = json.load(open(a.lengths))['lengths']
    os.makedirs(a.out, exist_ok=True)
    stats, outputs = {}, {}
    for split in ['train', 'val', 'test']:
        man = O.load_manifest(a.root, split)
        items = [(r.sample_id, os.path.join(a.root, r.csv), r.line, LEN) for r in man.itertuples()]
        with get_context('fork').Pool(a.procs) as pool:
            res = pool.map(work, items, chunksize=8)
        meta = man.set_index('sample_id')
        rows, lrows, pts, offs, uids = [], [], [], [0], []
        for sid, units, info, ext in res:
            m = meta.loc[sid]
            # rtl_rank = reading order by the unit's rightmost ink; writing_rank = order of the unit's first body stroke
            xmax = [max(ext[i][:, 0].max() for i in u['stroke_idx']) for u in units]
            rtl = np.argsort(-np.array(xmax), kind='stable'); rtl_rank = np.empty(len(units), int); rtl_rank[rtl] = np.arange(len(units))
            for k, u in enumerate(units):
                uid = f'{sid}_L{k:03d}'
                P = np.vstack([np.c_[ext[i], np.full(len(ext[i]), j)] for j, i in enumerate(u['stroke_idx'])])
                x0, y0 = P[:, 0].min(), P[:, 1].min(); x1, y1 = P[:, 0].max(), P[:, 1].max()
                rows.append(dict(unit_id=uid, sample_id=sid, split=split, writer_id=m.writer_id, writing_rank=k, rtl_rank=int(rtl_rank[k]),
                                 n_strokes=len(u['stroke_idx']), n_body_strokes=len(u['body_stroke_idx']), n_mark_strokes=len(u['mark_stroke_idx']),
                                 n_points=len(P), stroke_idx=json.dumps(u['stroke_idx']), body_stroke_idx=json.dumps(u['body_stroke_idx']),
                                 mark_stroke_idx=json.dumps(u['mark_stroke_idx']), aligned_text=u['text'], n_text_ligs=u['n_text_ligs'],
                                 x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1)))
                pts.append(P.astype(np.int64)); offs.append(offs[-1] + len(P)); uids.append(uid)
            lrows.append(dict(sample_id=sid, split=split, writer_id=m.writer_id, n_strokes=len(ext), n_units=len(units), n_text_ligs=info['n_text'],
                              n_skipped_text_ligs=len(info['skipped']), skipped=' '.join(info['skipped']), n_unaligned_strokes=len(info['extras']),
                              n_joined_units=sum(u['n_text_ligs'] > 1 for u in units), transcript=m.line))
        U = pd.DataFrame(rows); Lt = pd.DataFrame(lrows)
        allp = np.vstack(pts)
        assert np.abs(allp).max() < 2 ** 31
        f_csv = os.path.join(a.out, f'ligatures_v3_{split}.csv'); f_npz = os.path.join(a.out, f'ligatures_v3_{split}.npz'); f_lin = os.path.join(a.out, f'lines_v3_{split}.csv')
        U.to_csv(f_csv, index=False); Lt.to_csv(f_lin, index=False)
        np.savez_compressed(f_npz, points=allp.astype(np.int32), unit_offsets=np.array(offs, np.int64), unit_ids=np.array(uids),
                            columns=np.array(['x', 'y', 'time', 'pressure', 'stroke_local']))
        if split == 'test' and a.check_test:
            ref = json.load(open(a.check_test)); got = {sid: units for sid, units, _, _ in res}
            same = sum([u['stroke_idx'] for u in got[s]] == [u['stroke_idx'] for u in ref[s]['units']] for s in ref)
            assert same == len(ref), f'test units differ from reference on {len(ref) - same} lines'
            stats['test_matches_scored_v3'] = f'{same}/{len(ref)} test lines identical to the v3 units scored against the gold lines'
        stats[split] = dict(lines=len(Lt), writers=int(Lt.writer_id.nunique()), units=len(U), text_ligatures=int(Lt.n_text_ligs.sum()),
                            units_per_line_median=float(Lt.n_units.median()), single_stroke_units=round(float((U.n_strokes == 1).mean()), 4),
                            units_with_marks=round(float((U.n_mark_strokes > 0).mean()), 4), joined_units=int(Lt.n_joined_units.sum()),
                            skipped_text_ligatures=int(Lt.n_skipped_text_ligs.sum()), unaligned_strokes=int(Lt.n_unaligned_strokes.sum()),
                            lines_units_eq_text=round(float((Lt.n_units == Lt.n_text_ligs).mean()), 4), points=int(len(allp)))
        outputs[split] = {os.path.basename(f): dict(sha256=sha(f), bytes=os.path.getsize(f)) for f in (f_csv, f_npz, f_lin)}
        print(split, stats[split], flush=True)
    here = os.path.dirname(os.path.abspath(__file__))
    shutil.copy(os.path.join(here, 'ouhdl_ligatures.py'), os.path.join(a.out, 'ouhdl_ligatures.py'))
    import scipy
    prov = dict(
        dataset='OUHD-L ligature units, division v3.0',
        description='Ligature units (a maximal run of joined letters, including single isolated letters, with its own dots and marks) '
                    'derived from the OUHD-L v1.0 online strokes by per-line alignment of pen-down strokes to the transcript.',
        created_utc=datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z', seconds=round(time.time() - t0, 1),
        source=dict(name='OUHD-L v1.0', doi='10.5281/zenodo.20642162', local_root=a.root,
                    manifests_sha256={s: sha(os.path.join(a.root, f'{s}.csv')) for s in ['train', 'val', 'test']},
                    core_archive_sha256=open(os.path.join(a.root, 'core.sha256')).read().split()[0] if os.path.exists(os.path.join(a.root, 'core.sha256')) else None,
                    licence='see LICENSE in the dataset root'),
        method=dict(code=dict(ligseg_v3_py=sha(L3.__file__), ouhdl_adapter_py=sha(O.__file__), builder=sha(os.path.abspath(__file__)),
                              loader=sha(os.path.join(here, 'ouhdl_ligatures.py'))),
                    params=L3.PARAMS, letter_lengths=dict(file=os.path.basename(a.lengths), sha256=sha(a.lengths), fitted_on='train split only'),
                    stroke_segmentation='maximal runs of rows with pen_down == 1; rows with non-finite x/y dropped (as uhwr_baselines.ouhdl.load_raw_strokes)',
                    uses=['pen-down stroke recording order', 'line transcript'],
                    unit_rule_for_one_continuous_stroke_across_a_non_joiner='kept as one unit (ink-based); open decision, <author> notes such ligatures are linguistically separate'),
        validation=dict(gold='21 test lines, 354 units, labelled by <author> 15 Oct 2026 (labels artifact b234d282-fdac-43d1-9e02-4797d8be3e7a)',
                        exact_unit_f1=dict(v1=0.667, v2=0.742, v3=0.834), f1_v3_minus_v2_95ci=[0.035, 0.152],
                        pairwise='<author> blind A/B on 73 regions: v3 43, v2 22, neither 7, equal 1 (sign test p = 0.0125)',
                        note='test-split gold lines must not be used for tuning later versions'),
        checks=['every pen-down stroke assigned to exactly one unit (all lines, all splits)',
                'extended stroke loader equals uhwr_baselines.ouhdl.load_raw_strokes x/y on every stroke'] + ([stats['test_matches_scored_v3']] if 'test_matches_scored_v3' in stats else []),
        stats={k: v for k, v in stats.items() if k in ('train', 'val', 'test')}, outputs=outputs,
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, platform=platform.platform()),
        command=' '.join(sys.argv))
    json.dump(prov, open(os.path.join(a.out, 'PROVENANCE.json'), 'w'), indent=1, ensure_ascii=False, default=str)
    print('done', round(time.time() - t0, 1), 's')


if __name__ == '__main__':
    main()
