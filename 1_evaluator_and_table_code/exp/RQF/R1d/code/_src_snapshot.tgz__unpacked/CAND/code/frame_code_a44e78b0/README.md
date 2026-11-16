# frame_code — canonical-frame transforms, replay loader with 4k gate, frame sampling wrappers (uhwr, 28 Oct 2026)

Phase 0 (Mac-only, CPU) implementation of the C1 / C3 candidates and the K1 kill check of
`uhwr_method_candidates_2026-10-27.md`. Nothing here has been trained or sampled on a GPU; `smoke_report.json` lists
the CPU smoke runs (tiny unit counts — **smoke numbers are not results**).

Code tag: `CODE_TAG.txt` (sha8 over the python code + lib files; the job files reference `code_<TAG>`).
`SHA256SUMS.txt` covers every file; `frame_arm.sh` / `frame_k1.sh` refuse to run if it does not verify.

## Layout
| file | role |
|---|---|
| `e0_frames.py` | frames identity / mirror_x / vt on the stored 64×64 condition and the [x,y,pen,eos] sequence; exact inverse for sampled (N,L,4) arrays; pixel convention documented in the module docstring |
| `e0_dataset.py` | `OuhdlE0Dataset(frame=, frame_scripts=, script_tag=)` applies the frame in `__getitem__` to both condition and sequence (stored conditions are pre-rendered, so the frame acts on pixels, not a re-render) |
| `e0_replay.py` | `ReplayBatchSampler` (fixed Urdu share q_u, replay p_s ∝ n_s^(1/5), optional curriculum), `gate_eval` (K3 gate: gold 354 + 200 UNIPEN-de, uhwr_eval 0.3.1 native, Route A preds + sidecars) |
| `e0_train.py` | E0 trainer + M5 `--init-from` + `--frame` + `--replay-npz/--urdu-share[...]` + `--gate-step` (in-job) / `--gate-only`; `ReplayTrainer1D.train` = repo loop + gate hook |
| `frame_sample.py` | routeA_sample-style sampling of one checkpoint on sets × frames × seeds with the inverse applied; Route A file layout, frame in the sidecar |
| `frame_k1_summary.py` | K1 readout: per-frame means + writer-cluster bootstrap CIs, paired deltas vs identity, kill rule, loop-dominated subset |
| `e0_adapter.py`, `e0_sample.py`, `e0_score.py`, `m5_select.py` | unchanged copies (E0 code_4cd40f57 / M5 selection) so scoring + selection run from this directory |
| `lib/uhwr_eval` (0.3.1), `lib/uhwr_baselines` | pinned evaluator (put `lib` FIRST on PYTHONPATH) |
| `test_e0_frames.py` | T1–T6 tests (see smoke_report.json → `test_e0_frames`) |
| `jobs/frame_arm.sh`, `jobs/frame_k1.sh`, `jobs/frame_provenance.py` | platform-neutral pipelines (CLUSTER-A + CLUSTER-B call them); both ran end to end on the Mac CPU smoke (smoke_report.json: smoke_frame_arm_sh rc 0 with PROVENANCE written, smoke_frame_k1_sh) |
| `jobs/clusterA/*.sbatch`, `jobs/clusterB/*.sbatch`, `jobs/vps/*.cmd` | job files |

## Frame definitions (short; full text in e0_frames.py)
* mirror_x: x_n → −x_n on the trajectory; condition columns c → 64 − c (c = 1..63), column 0 := background.
* vt: y_n → −y_n and the point order reversed inside every pen-down stroke (pen/eos columns and stroke order unchanged);
  condition rows r → 64 − r, row 0 := background.
* Why the shifted flip and not `np.flipud`: uhwr_eval renders with cv2 at sub-pixel shift 4 (continuous y = r is the centre
  of row r), the sign flip is the only float32-bitwise reflection, and `r → 64 − r` is the pixel map consistent with it
  (measured: 89 k vs 1.78 M grey-level abs-diff against a re-render of the flipped ink on 50 units; test T3). Row/column 0
  is background on every rendered condition of every E0 split (T2), so the frame is a lossless involution there.
  `cond_dataset` (PNG crops) is not edge-clean on 554/4406 test units — frames are defined for the rendered condition.
* All three frames are involutions: inverse == forward (`inverse_frame_preds` before `e0_score.py`).

## Exact commands

### Unit tests (Mac, env uhwr-eval; `D` = experiments/e0/data)
```
PYTHONPATH=lib python test_e0_frames.py --npz $D/e0_val.npz --n 50 --all-splits $D/e0_train.npz $D/e0_test.npz $D/e0_gold.npz --out test_e0_frames_report.json
```

### K1 — frame kill check (LUO ms6 zero-shot; OUHD-L val250 + test 4,406 in identity/mirror_x/vt, IAM val250 in mirror_x + identity; seeds 0 1 2)
CLUSTER-A (A30, ~1–2 GPU-h sampling + CPU scoring in the same job):
```
B=$U; mkdir -p $B/exp/FRAME && tar -xzf frame_code_<TAG>.tar.gz -C $B/exp/FRAME   # -> $B/exp/FRAME/code_<TAG>
sbatch $B/exp/FRAME/code_<TAG>/jobs/clusterA/frame_k1_clusterA.sbatch
# outputs: $B/exp/FRAME/k1/preds/<set>__LUO_ms6_<frame>__seed<s>_preds.npz (+.json), *_rows.csv, k1_summary_<set>.json, summary_k1.csv, PROVENANCE_K1.json
```
VPS (runner queue): copy `jobs/vps/500_frame_k1_luo_ms6.cmd` to `$VPS_ROOT\uhwr\runner\queue\` after extracting the tarball to
`$VPS_ROOT\uhwr\code\frame_<TAG>\`; preds land in `$VPS_ROOT\uhwr\runs\FRAME\k1\preds\`; pull to CLUSTER-A (vps_pull) and score with
`frame_k1.sh` (set `PYG` to the CPU launcher; sampling is skipped for existing files).
Direct call (any platform):
```
python frame_sample.py --label LUO_ms6 --run-dir exp/M5/runs/luo --milestone 6 --stage <STAGE> --sets k1 --frames identity mirror_x vt --ltr-frames mirror_x identity --seeds 0 1 2 --out-dir <OUT> --batch-size 256
python e0_score.py e0 --npz <STAGE>/data/ouhdl_e0_v3/e0_val.npz --preds <OUT>/preds/val250_ouhdl__LUO_ms6_vt__seed0_preds.npz --method LUO_ms6_vt --input rendered --out <OUT>/preds/val250_ouhdl__LUO_ms6_vt__seed0_rows.csv --procs 14
python frame_k1_summary.py --rows "<OUT>/preds/val250_ouhdl__LUO_ms6_*__seed*_rows.csv" --set val250_ouhdl --out <OUT>/k1_summary_val250_ouhdl.json
```
val250 = indices 0..249 of e0_val.npz in stored order; the unit ids are written into each val250 sidecar (`unit_ids`).

### C3 — p01 arm with replay q_u = 0.25 and the 4k gate (control for C1; identity frame)
CLUSTER-A:
```
sbatch -J frame_c3_p01 --export=ALL,ARM=c3_p01_identity_replay,FRAC=p01,FRAME=identity,REPLAY=1 $B/exp/FRAME/code_<TAG>/jobs/clusterA/frame_arm_clusterA.sbatch
# ablations: QU=0.5 -> ARM=c3_p01_identity_replay50 ; curriculum: QU=0.2,QU_END=0.5,CUR_START=7000 -> ARM=c3_p01_identity_cur ; p10: FRAC=p10
```
CLUSTER-B: same `--export` with `jobs/clusterB/frame_arm_clusterB.sbatch`. VPS: `jobs/vps/510_frame_c3_p01_identity_replay_gate.cmd`
(511 = q_u 0.5, 512 = curriculum, 531 = p10); the .cmd trains, stops on GATE_FAIL, then samples val1000 for every milestone;
selection (`m5_select.py --valsets ouhdl1000`) and scoring run on CLUSTER-A after the pull; final sets via `540_frame_final_sets_TEMPLATE.cmd`.
Direct training call (what frame_arm.sh / the .cmd run; D = data/m5_multiscript_v1, OU = data/ouhdl_e0_v3, DE = data/d2_v2/unipen_de):
```
python e0_train.py --e0-train-npz $D/m5_ft_ouhdl_p01.npz --condition rendered --run-dir <RUN> --init-from exp/M5/runs/luo/checkpoints/checkpoint_000006.pt \
  --frame identity --frame-scripts ouhdl --replay-npz $D/m5_luo_train.npz --urdu-share 0.25 \
  --gate-step 4000 --gate-gold-npz $OU/e0_gold.npz --gate-unipen-npz $DE/e0_test.npz --gate-label c3_p01_identity_replay_ms4 \
  --batch-size 128 --gradient-accumulate-every 4 --learning-rate 8e-5 --train-steps 10000 --save-every 1000 --validation-every 1000 \
  --sequence-length 320 --image-size 64 --sampling-timesteps 50 --train-fraction 0.99 --num-workers 4 \
  --wandb-mode disabled --random-seed 20261016 --sample-count 25 --classes 0 --notes frame_c3_p01_identity_replay
```
Gate: after checkpoint_000004.pt is saved, `eval/gate_4k.json` = {gold_aiou, gold_tau, unipen_direction, pass, thresholds
(0.750 / 0.80 / 0.90), checks, files}; preds `eval/gold_ouhdl__<label>__seed0_preds.npz`, `eval/german_unipen_de_test__<label>__seed0_preds.npz`
(+ sidecars, rows csv). On failure the run dir gets `GATE_FAIL` (json) and the process exits 0; a restart sees GATE_FAIL and exits 0.
Standalone gate on an existing checkpoint:
```
python e0_train.py --e0-train-npz <any e0 npz> --run-dir <RUN> --gate-only --gate-milestone 4 --frame <frame of the arm> \
  --gate-gold-npz $OU/e0_gold.npz --gate-unipen-npz $DE/e0_test.npz --gate-label <ARM>_ms4 [--gate-checkpoint PATH]
```

### C1 — direction-canonical arms (frame chosen by K1; 2×2 at p01, plus p10)
```
# frame + replay            : --export=ALL,ARM=c1_p01_vt_replay,FRAC=p01,FRAME=vt,REPLAY=1
# frame only (no replay)    : --export=ALL,ARM=c1_p01_vt_noreplay,FRAC=p01,FRAME=vt,REPLAY=0
# replay only               : = the C3 arm above (identity, REPLAY=1)
# identity / none           : = the existing Route A LUO-FT p01 (not re-run here; ARM=luoft_p01_identity,FRAME=identity,REPLAY=0 reproduces it under this code)
# p10                       : --export=ALL,ARM=c1_p10_vt_replay,FRAC=p10,FRAME=vt,REPLAY=1
# mirror_x variant          : FRAME=mirror_x (VPS 522)
sbatch -J frame_c1_p01 --export=ALL,ARM=c1_p01_vt_replay,FRAC=p01,FRAME=vt,REPLAY=1 $B/exp/FRAME/code_<TAG>/jobs/clusterA/frame_arm_clusterA.sbatch
```
VPS: 520 (vt+replay), 521 (vt, no replay), 522 (mirror_x+replay), 530 (p10 vt+replay). The frame applies to Urdu units only
(`--frame-scripts ouhdl`); replay units stay identity. frame_arm.sh phase 3 samples test_ouhdl/gold_ouhdl in the arm's frame
(inverse applied) and german_unipen_de_test/val250_iam in identity, seeds 0 1 2, then scores and writes
`eval/PROVENANCE_frame_<ARM>.json` (init/urdu/replay npz sha256, selection, gate json, replay config + draw statistics, preds sha256).

### Smoke (CPU) — exactly what smoke_report.json records
```
# 20 optimiser steps on the 200-unit slice, eff. batch 8 (batch 4 x accum 2), curriculum 0.25->0.5 from step 10, gate at step 20 on 5+5 units
python e0_train.py --e0-train-npz m5_ft_ouhdl_p01.npz --run-dir RUN --frame vt --frame-scripts ouhdl --replay-npz m5_luo_train_slice200.npz \
  --urdu-share 0.25 --urdu-share-end 0.5 --curriculum-start-step 10 --batch-size 4 --gradient-accumulate-every 2 --train-steps 20 --save-every 10 \
  --validation-every 10 --sequence-length 320 --image-size 64 --sampling-timesteps 50 --train-fraction 0.99 --num-workers 0 --wandb-mode disabled \
  --random-seed 20261016 --sample-count 1 --classes 0 --no-amp --gate-step 20 --gate-gold-npz e0_gold.npz --gate-unipen-npz unipen_de/e0_test.npz \
  --gate-max-units 5 --gate-batch-size 8 --gate-label smoke_ms2
# same with --init-from luo_ms6_model_ema.pt (LUO weights) for 4 steps; frame_sample smoke: --max-units 8 on val250_ouhdl/val250_iam/gold_ouhdl
```

## Inputs this code expects on the target (CLUSTER-A-relative paths; VPS stage and CLUSTER-B mirror them)
`exp/M5/runs/luo/{run_config.json,checkpoints/checkpoint_000006.pt}` (sha256 ab6b178d…), `data/m5_multiscript_v1/{m5_luo_train.npz (690441b6…),
m5_ft_ouhdl_p01.npz (0e71475d…), m5_ft_ouhdl_p10.npz (3770d4d4…), m5_val_iam.npz (5ecbd8b0…)}`, `data/ouhdl_e0_v3/e0_{val,test,gold}.npz`,
`data/d2_v2/unipen_de/e0_test.npz (c5f1c29e…)`, nagamatsu repo src (7c5cf623) on PYTHONPATH, venv e0 (torch 2.7.1 + accelerate/ema_pytorch/einops/wandb/lmdb).
