# d2moe_code — D2 adapter experts + gate on the frozen M5 joint backbone (uhwr, 24 Oct 2026)

Backbone = Nagamatsu `Unet1D` with the **M5 joint ms9** EMA weights (27,092,036 parameters), frozen and kept in eval mode
(BatchNorm2d running stats, positional-encoding dropout off; parameter sha256 recorded at start, at every checkpoint and
at the end — `backbone_unchanged` must be true). Experts = LoRA adapters (rank 16, alpha 16, B zero-initialised so every
expert starts as a zero delta) on the attention projections: `CrossAttention.to_q/to_kv/to_out[0]` at every level + mid +
final, mid `Attention.to_qkv/to_out`, `LinearAttention.to_qkv/to_out[0]` at every level — 75 wrapped modules,
**953,344 parameters per expert**. Gate = 2-layer MLP (512 -> 128 -> E, SiLU, zero-initialised output layer, 66,051
parameters at E=3) on the global-average-pooled deepest ImageEncoder map (4x4x512) of the condition image -> softmax over
experts; `--gate-feat all` uses the concatenated pooled maps (1024-d). Forward = backbone with the expert-weighted adapter
deltas `sum_e g_e B_e A_e x`. Loss = Nagamatsu denoising loss + `--lb-coef` x Switch load-balance penalty
`E * sum_e f_e P_e`. Only adapters + gate train (2,926,083 parameters at E=3). EMA of the trainable parameters follows
ema_pytorch (0.995, every 10, after step 100, warm-up `1-(1+n)^(-2/3)`) and is what `d2moe_sample.py` uses.

**Stage 1** (`--stage 1`): E=3 experts + gate on casia_zh / casia_en / hpl_tamil / iam from `m5_joint_train.npz`.
**Stage 2** (`--stage 2 --init-from-stage1 CKPT`): loads the stage-1 EMA adapters/gate, adds one expert `rtl` whose
`(A, B)` is the best rank-16 (SVD) approximation of the gate-weighted mix `sum_e w_e B_e A_e` with `w` = mean stage-1 gate
on `--gate-init-sample 256` Urdu training units, and a gate row = the same mix of the existing rows; trains on a Urdu
subset (`m5_ft_ouhdl_p01/p10/p30.npz` with `--script-override ouhdl`, `data/ouhdl_e0_v3/e0_train.npz` for p100, or
`--urdu-frac p` of any file) with the old experts frozen (`--old-experts frozen`, default) or trainable; gate trainable
unless `--no-train-gate`. `--new-expert-init fresh` gives a zero-delta new expert instead.

Checkpoints (`checkpoints/checkpoint_<ms>.pt`, ~15–20 MB) every `--save-every` steps: `trainable`, `ema_trainable`,
`ema` (state), `opt`, `scaler`, `step`, `config` (experts, rank, targets, gate, backbone file + parameter sha256,
stage-2 init record), `backbone_sha256_now/at_start`, code sha, GPU. Auto-resume from the latest one (seed = seed + step;
adapter/gate architecture is read from the checkpoint). At every validation: `gate_weights/step_<step>.json` (mean gate
vector + argmax histogram per script) and a row per (step, script, expert) in `gate_log.csv`. Per-step log
(`train_log.jsonl`): `loss`, `loss_den`, `loss_lb`, `grad_norm`, `routing` (argmax counts per expert in the step),
`gate_by_script` (mean gate per script), `peak_rss_mb`, `cuda_max_alloc_gb`. `run_config.json`: arguments, config, code
sha256, params, train npz sha256, GPU, torch/CUDA, hostname, Slurm id, Urdu-subset rule.

Files: `d2moe_train.py`, `d2moe_model.py` (MoELoRA, Gate, MoEUnet1D, load_balance_loss), `d2moe_sample.py` (DDIM sampler;
output = e0_sample.py format + `scripts`, `gate` (N,E), `expert_names`; `--force-expert NAME` for oracle-expert runs),
`uhwr_route_common.py`, `jobs/`, `smoke/`, `SHA256SUMS.txt`. Requires the Nagamatsu repo `src/` (7c5cf623) on `PYTHONPATH`.

## Unpack

`<sha8>` = first 8 hex digits of the tarball's sha256 (`d2moe_code_<sha8>.tar.gz`).
```
# CLUSTER-A
cd $U/exp/D2_MOE/code && mkdir -p d2moe_code_<sha8> && tar -xzf d2moe_code_<sha8>.tar.gz --strip-components=1 -C d2moe_code_<sha8> && (cd d2moe_code_<sha8> && sha256sum -c --quiet SHA256SUMS.txt && echo code-ok)
# CLUSTER-B: same with $U_B/exp/D2_MOE/code
# VPS
mkdir $VPS_ROOT\uhwr\code\d2moe_code && tar -xzf $VPS_ROOT\uhwr\code\d2moe_code_<sha8>.tar.gz --strip-components=1 -C $VPS_ROOT\uhwr\code\d2moe_code
```

## VPS (RTX 4090, Windows queue runner)

`PYTHONPATH=$VPS_ROOT\uhwr\winshim;$VPS_ROOT\uhwr\repos\nagamatsu\src;$VPS_ROOT\uhwr\code\d2moe_code`, python
`C:\ProgramData\miniconda3\envs\uhwr\python.exe`, inputs under `$VPS_ROOT\uhwr\stage\...`, `--num-workers 0`.
Queue `jobs\vps_d2_stage1.cmd` as `runner\queue\NNN_d2_stage1.cmd`, then `jobs\vps_d2_stage2_p10.cmd` (edit `S1`, the
train npz and `RUN` for p01 / p30 / p100 and for `--old-experts trainable`), sampling `jobs\vps_d2_sample.cmd`.
Memory: backbone fp32 + adapters, micro 64 — below the E0 micro-64 figure (6.5 GiB) since only adapters carry Adam state.

Stage 1 command (inside the job file):
```
%PY% %CODE%\d2moe_train.py --stage 1 --run-dir $VPS_ROOT\uhwr\runs\D2_MOE\stage1_e3r16 ^
  --backbone %STAGE%\exp\M5\runs\joint\checkpoints\checkpoint_000009.pt --backbone-weights ema ^
  --train-npz %D%\m5_joint_train.npz --scripts casia_zh casia_en hpl_tamil iam ^
  --val-npz casia_zh=%D%\m5_val_casia_zh.npz casia_en=... hpl_tamil=... iam=... ouhdl=%D%\m5_val_ouhdl.npz ^
  --n-experts 3 --rank 16 --lora-alpha 16 --lora-targets cross,attn,linear --gate-hidden 128 --gate-feat deepest --lb-coef 0.01 ^
  --batch-size 64 --grad-accum 4 --train-steps 20000 --save-every 1000 --val-every 1000 --learning-rate 1e-4 --warmup-steps 500 ^
  --keep-last 5 --keep-every 5000 --num-workers 0 --seed 20261024
```
Stage 2 (p10, old experts frozen):
```
%PY% %CODE%\d2moe_train.py --stage 2 --run-dir $VPS_ROOT\uhwr\runs\D2_MOE\stage2_p10_frozen ^
  --init-from-stage1 $VPS_ROOT\uhwr\runs\D2_MOE\stage1_e3r16\checkpoints\checkpoint_000020.pt ^
  --backbone %STAGE%\exp\M5\runs\joint\checkpoints\checkpoint_000009.pt --backbone-weights ema ^
  --train-npz %D%\m5_ft_ouhdl_p10.npz --script-override ouhdl --scripts ouhdl --val-npz ouhdl=%D%\m5_val_ouhdl.npz casia_zh=... casia_en=... hpl_tamil=... iam=... ^
  --new-expert-name rtl --new-expert-init gate_mix --gate-init-sample 256 --old-experts frozen --train-gate --lb-coef 0.01 ^
  --batch-size 64 --grad-accum 4 --train-steps 10000 --save-every 1000 --val-every 1000 --learning-rate 1e-4 --warmup-steps 200 ^
  --keep-last 5 --keep-every 5000 --num-workers 0 --seed 20261024
```
Sampling: `%PY% %CODE%\d2moe_sample.py --run-dir RUN --milestone latest --backbone %STAGE%\exp\M5\runs\joint\checkpoints\checkpoint_000009.pt --npz NPZ --sampling-timesteps 50 --ddim-eta 0 --batch-size 256 --seed 0 --out OUT.npz`

## CLUSTER-A (Slurm, apptainer `images/torch27.sif`, `venvs/e0`)

```
cd $U && mkdir -p exp/D2_MOE/logs exp/D2_MOE/runs
TAG=<sha8> STAGE=1 sbatch --export=ALL,TAG,STAGE exp/D2_MOE/code/d2moe_code_<sha8>/jobs/clusterA_d2_stage.sbatch
TAG=<sha8> STAGE=2 P=p10 OLD=frozen S1=$U/exp/D2_MOE/runs/stage1_e3r16/checkpoints/checkpoint_000020.pt \
  sbatch --export=ALL,TAG,STAGE,P,OLD,S1 exp/D2_MOE/code/d2moe_code_<sha8>/jobs/clusterA_d2_stage.sbatch
```
(gpuidle, 1 GPU, `--exclude=gpu[001-002]`, 8 CPUs, 64 GB, 1 day, `--requeue`; P in p01 / p10 / p30 / p100; run dirs
`exp/D2_MOE/runs/stage1_e3r16`, `exp/D2_MOE/runs/stage2_<P>_<OLD>`.)

## CLUSTER-B <site> (Pyxis)

```
sbatch --parsable -A <account> -p H100-RP,H100 --gpus=1 --cpus-per-task=8 --mem=64G --time=1-00:00:00 -J d2moe_s1 \
  -o $U_B/exp/D2_MOE/clusterB/jobs/s1_%j.out --requeue --export=ALL,TAG=<sha8>,STAGE=1 \
  $U_B/exp/D2_MOE/code/d2moe_code_<sha8>/jobs/clusterB_d2_stage.sh
# stage 2: --export=ALL,TAG=<sha8>,STAGE=2,P=p10,OLD=frozen,S1=$U_B/exp/D2_MOE/clusterB/runs/stage1_e3r16/checkpoints/checkpoint_000020.pt
```
Run dirs under `exp/D2_MOE/clusterB/runs/`; exit 8 when the 1-day limit ends the job — resubmit to resume.

## Mac CPU smoke (24 Oct 2026, `smoke/smoke_report.json`)

- stage 1, 400 units (4 scripts x 100), micro 8 x accum 2, 30 steps: losses finite; `loss_den` 0.0021–0.0059, `loss_lb`
  1.000–1.003; per-step argmax routing counts logged (uniform gate at step 0 -> ties; step 15 `{'expert0': 1, 'expert1': 9, 'expert2': 6}`);
  fixed-noise val den casia_zh 0.0073 -> 0.0071, iam 0.0028 -> 0.0028; mean gate per script written at steps 0/15/30;
  **backbone sha256 identical before and after** (`backbone_unchanged: true`); peak RSS 5.9 GB; 4.4 s/step CPU.
- fixed-batch check (lr 1e-3, 20 steps): den 0.0075 -> 0.0012, monotone after step 2.
- stage 2 from the stage-1 checkpoint: expert `rtl` added from mix {0.3332, 0.3334, 0.3334}; trainable tensors 154
  (new expert 150 + gate 4), 1,019,524 trainable parameters; 12 steps on 50 Urdu units (p = 0.5 rule recorded); backbone unchanged;
  resume of the stage-2 run dir: `resumed milestone 2 step 12`, 4 experts restored.
- kill-and-resume (stage 1): killed after step 12 -> `resumed milestone 2 step 10 seed 10`, ran 11..20.
- sampling: `d2moe_sample.py` DDIM-50, 8 OUHD-L units, stage-2 EMA adapters: finite, per-unit gate saved, mean Chamfer to GT ink 0.34 px.
- backbone from a full Trainer1D checkpoint (format `trainer1d`) tested as well as from the fp16 extract.

Deviations from the spec: the gate's pooled features are taken from the frozen backbone encoder (no separate image
encoder); the EMA is a parameter-dict EMA replicating ema_pytorch's schedule (not the `EMA` module); the new stage-2
expert is the rank-16 SVD truncation of the mixed delta (exact when the mix has rank <= 16); `--warmup-steps` added.
