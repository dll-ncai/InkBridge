"""FMDX REPA targets: frozen FM gate-layer tokens on the shared training subset of the LUO backbone mix.

For one FM (entry of fm_registry.json: screen extractor module(s) registered with fmd_harness extract_api, pre-named gate
layer, extractor kwargs): read exp/FMD/xs/verdicts/verdict_<fm>.json first - status FAIL -> write SKIPPED_GATE_FAIL.json and
exit 0 (nothing extracted); otherwise extract the gate layer for the subset units in chunks (screen extractor objects,
unedited), resample the native grid to 16x16 by the pre-registered rule (extract_api.resample_grid: bilinear,
align_corners=False, antialias=True; identity if already 16x16), store float16 tokens row-major (r, c) = the student patch
order (M,256,D). Shards (--shard k --n-shards n) extract chunks c with c % n == k into <out>.partial/chunk_<c>.npy; the shard that
finds every chunk present merges them (O_EXCL lock) into the final .npy. Requeue-safe: finished chunks are skipped.
Inputs: the 224-px render array of the LUO train units (prep_renders.py seqset, npz order) + the subset positions (.npy,
rows of m5_luo_train.npz, fixed by the capacity note). Images are passed to the extractor exactly as the screen passed
them: uint8 (N,224,224) black ink on white.
Screen-reproduction check: the same extractor re-extracts the first --repro-n units of val1000 (exp/FMD/data/
fmd_render224_val1000.npz) and of each available cross-script screen set, compared with the cached screen features of the
gate layer when they exist (max |diff|, min cosine). Output files:
  <out_dir>/<fm>/repa_<fm>_<layer>_G16__<subset>.npy  + __unit_ids.npy + .meta.json (fmd_meta/1) + DONE.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from fmd_harness import common as C
from fmd_harness.extract_api import get_extractor, resample_grid, feature_path

XS = C.ROOT / "exp/FMD/xs"


def read_status(fm, vdir):
    p = Path(vdir) / f"verdict_{fm}.json"
    if not p.is_file():
        return "MISSING", None
    try:
        return str(json.loads(p.read_text()).get("status", "PENDING")).upper(), C.sha256_file(p)
    except Exception:
        return "UNREADABLE", None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fm", required=True)
    ap.add_argument("--registry", default=str(Path(__file__).resolve().parent / "fm_registry.json"))
    ap.add_argument("--render-npy", required=True)
    ap.add_argument("--render-unit-ids", required=True)
    ap.add_argument("--subset-pos", required=True, help=".npy int64 rows of the render array (capacity-note subset)")
    ap.add_argument("--subset-name", required=True)
    ap.add_argument("--out-dir", default=str(XS / "targets"))
    ap.add_argument("--verdict-dir", default=str(XS / "verdicts"))
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--repro-n", type=int, default=8)
    ap.add_argument("--max-units", type=int, help="smoke only")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n-shards", type=int, default=1)
    a = ap.parse_args()
    regall = json.loads(Path(a.registry).read_text())
    for d in regall.get("_pythonpath", []):
        if d not in sys.path:
            sys.path.append(d)
    imported = {}
    for m in regall.get("_import", []):           # every screen extractor module (registers into fmd_harness 1.0.1, unedited)
        try:
            importlib.import_module(m); imported[m] = "ok"
        except Exception as e:  # noqa: BLE001
            imported[m] = f"{type(e).__name__}: {str(e)[:200]}"
    print("extractor modules:", imported, flush=True)
    reg = regall.get("fms", {}).get(a.fm, {})
    kw = dict(reg.get("kw", {}))
    od = Path(a.out_dir) / a.fm; od.mkdir(parents=True, exist_ok=True)
    st, vsha = read_status(a.fm, a.verdict_dir)
    if st == "FAIL":
        C.write_json_atomic(od / "SKIPPED_GATE_FAIL.json", dict(fm=a.fm, verdict_status=st, verdict_sha256=vsha, utc=C.utc_now(), slurm=os.environ.get("SLURM_JOB_ID")))
        print(f"{a.fm}: verdict FAIL -> skipped"); return
    base = f"repa_{a.fm}_gate_G16__{a.subset_name}"
    out = od / f"{base}.npy"
    if (od / f"{base}.DONE.json").exists():
        print(f"{out} already done"); return
    pos = np.load(a.subset_pos).astype(np.int64)
    if a.max_units:
        pos = pos[: a.max_units]
    X = np.load(a.render_npy, mmap_mode="r")
    uids = np.load(a.render_unit_ids, allow_pickle=False).astype(str)
    assert X.shape[0] == len(uids)
    ex0 = get_extractor(a.fm, **kw)
    layer = reg.get("layer") or getattr(ex0, "gate_layer", None) or "last"
    print(f"{a.fm}: gate layer {layer} ({'registry' if reg.get('layer') else 'extractor attribute'})", flush=True)
    part = od / f"{base}.partial"; part.mkdir(exist_ok=True)
    n_chunks = (len(pos) + a.chunk - 1) // a.chunk
    mine = [c for c in range(n_chunks) if c % a.n_shards == a.shard]
    t0 = time.time()
    ex = None
    for c in mine:
        cf = part / f"chunk_{c:05d}.npy"
        if cf.exists():
            continue
        if ex is None:
            ex = ex0; ex.setup()
        sl = pos[c * a.chunk:(c + 1) * a.chunk]
        feats = ex.extract(np.asarray(X[sl], np.uint8))[layer]
        G = resample_grid(feats, 16) if feats.shape[1] != 16 or feats.shape[2] != 16 else np.asarray(feats, np.float32)
        G = np.asarray(G, np.float32)
        assert np.isfinite(G).all(), f"non-finite features in chunk {c}"
        tmp = part / f"chunk_{c:05d}.tmp.npy"
        np.save(tmp, G.reshape(len(sl), 256, G.shape[3]).astype(np.float16)); os.replace(tmp, cf)
        print(f"shard {a.shard}/{a.n_shards} chunk {c + 1}/{n_chunks} {time.time() - t0:.0f}s", flush=True)
    done = sorted(part.glob("chunk_?????.npy"))
    if len(done) < n_chunks:
        print(f"shard {a.shard}: {len(done)}/{n_chunks} chunks present; merge left to the last shard"); return
    try:
        fd = os.open(str(part / "MERGE.lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
    except FileExistsError:
        print("another shard is merging"); return
    D = np.load(done[0], mmap_mode="r").shape[2]
    tp = part / "tokens.npy"
    M = np.lib.format.open_memmap(tp, mode="w+", dtype=np.float16, shape=(len(pos), 256, D))
    for c in range(n_chunks):
        M[c * a.chunk:(c + 1) * a.chunk] = np.load(part / f"chunk_{c:05d}.npy")
    M.flush(); del M
    os.replace(tp, out)
    np.save(od / f"{base}__unit_ids.npy", uids[pos])
    if ex is None:
        ex = ex0; ex.setup()
    T = np.load(out, mmap_mode="r")
    # screen reproduction: first --repro-n units of each screen set vs the cached screen features of the gate layer
    repro = {}
    sets = {"ouhdl_val1000": C.FMD / "data/fmd_render224_val1000.npz"}
    for sn in ("casia_zh_val250", "casia_en_val250", "hpl_tamil_val250", "iam_val250"):
        sets[sn] = XS / "data" / f"xs_render224_{sn}.npz"
    for sname, sp in sets.items():
        refs = [XS / "features" / a.fm / sname / f"{layer}__{sname}.npy"]
        if sname == "ouhdl_val1000":
            refs.append(C.FMD / "features" / a.fm / f"{layer}__val1000.npy")
        ref_p = next((r for r in refs if r.exists()), None)
        if ref_p is None or not Path(sp).exists():
            repro[sname] = dict(status="no cached screen features", tried=[str(r) for r in refs]); continue
        V = np.load(sp, allow_pickle=False)["img224"][: a.repro_n]
        rep = ex.extract(V)[layer]
        rep = resample_grid(rep, 16) if rep.shape[1] != 16 else rep
        rep = np.asarray(rep, np.float32).astype(np.float16).astype(np.float32)
        ref = np.load(ref_p, mmap_mode="r")[: a.repro_n].astype(np.float32)
        if ref.shape[1] != 16:
            ref = resample_grid(ref, 16)
        repro[sname] = dict(n=a.repro_n, max_abs=float(np.abs(rep - ref).max()), ref_max_abs=float(np.abs(ref).max()),
                            cos_min=float(np.min(np.sum(rep * ref, -1) / (np.linalg.norm(rep, axis=-1) * np.linalg.norm(ref, axis=-1) + 1e-12))),
                            ref=str(ref_p))
    info = dict(fm=a.fm, layer=layer, n_units=int(len(pos)), target_shape=list(T.shape), dtype="float16",
                grid_rule="fmd_harness extract_api.resample_grid (bilinear, align_corners=False, antialias=True) native -> 16 (identity if 16)",
                token_order="row-major (r, c) of the 16x16 grid = student patch order", extractor_record=ex.record() if hasattr(ex, "record") else None,
                seconds=round(time.time() - t0, 1), units_per_s_merging_shard=round(len(mine) * a.chunk / max(1e-9, time.time() - t0), 2), n_shards=a.n_shards,
                unit_ids_sha256=C.sha256_bytes("\n".join(uids[pos]).encode()), tokens_sha256=C.sha256_array(np.asarray(T)),
                mean_token_norm=float(np.linalg.norm(np.asarray(T[: min(512, len(pos))], np.float32), axis=-1).mean()),
                screen_repro=repro, verdict_status_at_start=st, verdict_sha256=vsha)
    C.write_meta(out, kind="fmdx_repa_targets", inputs={"render_npy": a.render_npy, "render_unit_ids": a.render_unit_ids, "subset_pos": a.subset_pos,
                 "registry": a.registry}, params=dict(fm=a.fm, layer=layer, extractor_modules=imported, kw=kw, subset=a.subset_name, chunk=a.chunk, max_units=a.max_units,
                 gate_layer=True, prereg="FMD_PREREG_2026-11-03.md section 6 + A1 (+ A3/A4)"),
                 extra=dict(result=info, script=dict(path=str(Path(__file__).resolve()), sha256=C.sha256_file(__file__))))
    C.write_json_atomic(od / f"{base}.DONE.json", dict(out=str(out), sha256=C.sha256_file(out), unit_ids=str(od / f"{base}__unit_ids.npy"), utc=C.utc_now()))
    print(json.dumps({k: info[k] for k in ("fm", "layer", "target_shape", "seconds", "n_shards", "screen_repro")}, default=str))


if __name__ == "__main__":
    main()
