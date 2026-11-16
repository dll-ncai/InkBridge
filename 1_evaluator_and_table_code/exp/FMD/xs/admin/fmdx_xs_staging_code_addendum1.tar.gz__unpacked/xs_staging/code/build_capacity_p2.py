#!/usr/bin/env python3
"""Part 2 of CAPACITY_NOTE.md: subset proposal, live state, lane assignment, arm priority.  Run after build_capacity_p1.py."""
import json, glob, os, collections, datetime
X = "$U/exp/FMD/xs"
exec(open(X + "/code/build_capacity_p1.py").read().split("L = []; A = L.append")[0])     # reuse loaders/helpers (T, tags, rate, gpuh, fmt, SET_*)
LS = json.load(open(X + "/admin/live_state_2026-11-03.json"))
meta = json.load(open(X + "/data/luo_repa_order_v1.csv.meta.json"))["result"]
L = []; A = L.append
A("## 4. Proposed fixed shared training subset for REPA targets"); A("")
A(f"Ordering file: `exp/FMD/xs/data/luo_repa_order_v1.csv` (80,000 rows; sha256 + code sha in `.meta.json`; unit_ids sha256 `{meta['unit_ids_sha256']}`).  Rule: per script, rows in file order are permuted by `default_rng([20261103, 7, i])` (scripts sorted: casia_en, casia_zh, hpl_tamil, iam); `rank_in_script` = position; subset of size N = all rows with `rank_in_script < N/4`; nested (10k within 20k within 40k within 80k); rows of the csv are interleaved by rank, so every csv prefix is script-stratified.")
A(""); A("Writers (sample_ids) covered by each prefix: " + "; ".join(f"N={k}: {v}" for k, v in meta["writers_in_prefix"].items()) + " (of 1,597 in the full mix).")
A("")
ref = "a4000"
A(f"Budget test, reference GPU = {ref} (the only type measured on all FMs at writing; slower than A100/H100, so a conservative bound).  Sum of GPU-h over the FMs of each set, one pass over N units:"); A("")
A("| N units (per script) | priority set (5 FMs) | all roster candidates with a rate (" + str(len([f for f in SET_ALL if rate(f, ref)])) + " FMs) |"); A("|---|---|---|")
choice = None
for N in (10000, 20000, 40000, 80000):
    sp = sum(gpuh(f, ref, N) or 0 for f in SET_PRI); sa = sum(gpuh(f, ref, N) or 0 for f in SET_ALL)
    A(f"| {N:,} ({N//4:,}) | {sp:.1f} GPU-h | {sa:.1f} GPU-h |")
    if sa <= 26.0: choice = N
A("")
A(f"**Proposal: N = {choice:,} units ({choice//4:,} per script, first {choice//4:,} ranks of `luo_repa_order_v1.csv`)** - the largest nested size whose all-candidate sum is within ~8% of the 24 GPU-h target (<= 26 GPU-h) on the slowest measured GPU (A4000); the next smaller nested size is the fallback if the sum must stay strictly <= 24.  Faster GPUs (A30/A100/H100) reduce the sum.  Where the priority set alone is the target, N = 80,000 costs the figure in the table above.  Cost drivers on the A4000 (units/s): sd21-dift-gate {fmt(rate('sd21-dift-gate', ref))}, cleandift-sd21 {fmt(rate('cleandift-sd21', ref))}, cleandift-sd15 {fmt(rate('cleandift-sd15', ref))}; every ViT-L-class FM is >= 30 units/s.".replace("{fmt(rate('sd21-dift-gate', ref))}", fmt(rate('sd21-dift-gate', ref))).replace("{fmt(rate('cleandift-sd21', ref))}", fmt(rate('cleandift-sd21', ref))).replace("{fmt(rate('cleandift-sd15', ref))}", fmt(rate('cleandift-sd15', ref))))
A(""); A("Storage of the 16 x 16 mapped gate-layer tokens (prereg grid rule), fp16, for the proposed N (native diffusion maps 48 x 48 x 1280 would be 9x larger and are not stored):"); A("")
A("| FM | channels | MB per unit | GB for N |"); A("|---|---|---|---|")
tot_gb = 0
for fm in SET_PRI + [f for f in SET_ALL if f not in SET_PRI]:
    if fm not in T: continue
    r = next(iter(T[fm].values())); chs = r.get("channels") or {}
    c = chs.get("last") or chs.get("t261_up1") or chs.get("up1") or (list(chs.values())[0] if chs else None)
    if not c: continue
    mb = 256 * c * 2 / 1e6; gb = mb * choice / 1e3; tot_gb += gb
    A(f"| {fm} | {c} | {mb:.2f} | {gb:.1f} |")
A(f"| **sum** | | | **{tot_gb:.0f}** |"); A("")
A("Dependency: LUO training renders at 224 px must be derived from the stored sequences (section 1); they are not byte-identical to the stored 64-px conditions, so a student/backbone trained on derived renders sees a slightly different image than the LUO backbone saw.")
A("")
A("## 5. Live cluster state"); A("")
A(f"Snapshot taken {LS['captured_utc']} (CLUSTER-A Slurm prints local time CET = UTC+1; CLUSTER-B Slurm prints CET as well; the VPS clock runs at UTC-8).  Raw text: `admin/live_state_2026-11-03.json`.")
A("")
A("### CLUSTER-A (account <user>)"); A("")
A("- `squeue` for the account at capture: only CPU jobs (`cf2_scor`, `cf2_watc` on skylake-96, other track) and this track's staging jobs; 0 GPU jobs running or pending -> 0 of the 16-GPU QoS cap in use at capture (held/confirmatory jobs of other tracks: none visible in squeue).")
A("- b200idle: single node <node> (8 x B200), state MIXED+DRAIN, reason 'Reboot' set 2026-11-02T14:57:49 (local), 4 of 8 GPUs allocated to other users.  `sbatch --test-only` for 1 x and 2 x B200 returned 'Requested node configuration is not available'.")
A("- A100: <node> IDLE+DRAIN (reason 'CUDA-Update', 2026-10-26), <node> ALLOCATED+DRAIN (reason 'Reboot', 2026-11-02): `--test-only` for 1 x and 2 x A100 (with and without `-p gpuidle`) returned 'Requested node configuration is not available'.")
A(""); A("| request (sbatch --test-only) | estimated start (local CET) |"); A("|---|---|")
for k, v in LS["clusterA"]["test_only"].items():
    if k.startswith("CLUSTER-B"): continue
    A(f"| {k} | {v.split('to start at')[1].split(' a ')[0].strip() if 'to start at' in v else v} |")
for k, v in LS.get("clusterA_more_test_only", {}).items():
    if k.startswith("CLUSTER-A"): A(f"| {k} | {v.split('to start at')[1].split(' a ')[0].strip() if 'to start at' in v else v} |")
A(""); A("A4000 timing job 24842562 was submitted at ~14:45 local with an estimate of 17:09 and started after minutes; estimates for gpuidle slots are therefore upper bounds.  gpuidle jobs are preemptible (three of this track's CPU staging jobs were preempted once on <node>).")
A(""); A("### CLUSTER-B (user <author>)"); A("")
A("| request (sbatch --test-only) | estimated start (local CET) |"); A("|---|---|")
for k, v in {**LS["clusterB"]["test_only"], **{kk: vv for kk, vv in LS.get("clusterA_more_test_only", {}).items() if kk.startswith("CLUSTER-B")}}.items():
    A(f"| {k.replace('CLUSTER-B ','')} | {v.split('to start at')[1].split(' a ')[0].strip() if 'to start at' in v else v} |")
A(""); A("Normal partitions (A100-80GB, H100, H200) are estimated to start in Dec 2026 - May 2027 (queue priority of this account); the `-RP` partitions (A100-RP, H100-RP) and A100-PCI/H200-SDS start within hours-days.  `-RP` jobs can be preempted: arms there must be requeue-safe.")
A(""); A("### VPS RTX 4090 (read-only)"); A(""); A("- " + LS["vps"])
A("- Number ranges: 400-406 are used by the cf2 track; free for FMDX: 440-499 (and 407-409).  The sealed-read cmds 050/300-349 are not in the queue directory any more (jobs.csv shows them done).")
A("")
A("## 6. Lane assignment"); A("")
A("| workload | primary lane | race / secondary lane | facts behind the choice |"); A("|---|---|---|---|")
A("| screen extraction of the new FMs on the 6 screen sets (val1000 + 4 x 250 + 3,927 = 5,927 units per FM) | CLUSTER-A gpuidle array (A4000/A30/A40; cap <= 6 GPUs) | CLUSTER-B A100-RP / H100-RP for FMs >= 7 GB (PE-Spatial-G14, C-RADIOv3-g, SigLIP2-giant, CleanDIFT / DIFT) | A4000 rates in section 2: 5,927 units take ~1.8 min (DINOv3-L, 56 u/s), ~12 min (PE-Spatial-G14, 8.4 u/s) and ~82 min (sd21-dift-gate, 1.2 u/s) on one A4000; A4000 slots started within minutes today |")
A("| REPA-target extraction on the fixed subset (section 4) | CLUSTER-A gpuidle arrays, one job per FM | CLUSTER-B A100-RP / H100-RP copies of the two diffusion FMs (largest GPU-h) | diffusion FMs hold 90%+ of the GPU-h in section 3 |")
A("| backbone arms (2 GPUs, 100k steps; control, DINOv3, CleanDIFT/SD2.1, PE-init, ...) | CLUSTER-B H100-RP (2 x and 4 x estimated within hours, see table) and A100-RP | CLUSTER-A gpuidle 2 x A30 (est. 4 Nov 06:13) / 2 x V100 / 2 x H100 (est. 5 Nov 13:56); b200idle only after <node> leaves DRAIN | B200 and A100 are drained at CLUSTER-A; CLUSTER-B `-RP` is preemptible -> every arm must checkpoint and resume; wall time per 100k steps on H100/A100/A30 has NOT been measured in this track (LUO on 2 x B200: ~15 h) |")
A("| FT (p01 / p10 / p30 / p100) + sampling on 7 test sets | VPS RTX 4090 (cmds 440-499) | CLUSTER-A gpuidle A30/A4000 race copies | 4090 queue empty except cf2 job 406 at capture |")
A("| transfers CLUSTER-A <-> CLUSTER-B | VPS hub from CLUSTER-B login1 / CLUSTER-A <node> (sha256 both ends) | CLUSTER-B direct download of public weights (sha-checked against CLUSTER-A manifest) | hub path was blocked 13:35-15:05Z by the VPS `from=` rule (restored by the lead with the user's approval) |")
A("")
A("## 7. Initial arm priority order"); A("")
A("1. **Control** (no-FM backbone arm): no target extraction; first to launch so its 100k steps are not blocked.")
A("2. **DINOv3-L/16** (only FM that passed the Urdu gate): REPA targets for N units cost "
  + (fmt(gpuh('dinov3-vitl16', ref, choice)) if choice else 'n/a') + " GPU-h on an A4000; weights stay on CLUSTER-A (licence), so CLUSTER-B copies can only receive target files.")
A("3. **CleanDIFT SD2.1** and the SD2.1 DIFT gate layer (largest Urdu order deltas in FMD, gate-failed on T0 shape only): targets cost "
  + (fmt(gpuh('cleandift-sd21', ref, choice)) if choice else 'n/a') + " (CleanDIFT, full-UNet load-test pass) and " + (fmt(gpuh('sd21-dift-gate', ref, choice)) if choice else 'n/a') + " (DIFT gate) GPU-h on an A4000 at the proposed N.")
A("4. **PE-init** (PE-Spatial / PE-Core encoder as initialisation of the 224-px encoder; PE-Spatial-L14 gate-layer targets "
  + (fmt(gpuh('pe-spatial-l14-448', ref, choice)) if choice else 'n/a') + " GPU-h; PE-Spatial-B16-512 and PE-Core-B16-224 have 14 x 14 maps and a ViT-B body).")
A("5. Others in the order of their Urdu / cross-script screen strength once verdict files exist (`exp/FMD/xs/verdicts/`): candidates already staged and timed are C-RADIOv3-{B,L,H,g}, C-RADIOv4-{H,SO400M}, PE-Spatial-{T,S,B,L,G}, SigLIP2-giant-opt-256, DINOv3 {ViT-S,B,H+,7B, ConvNeXt}, TIPSv2 {B,L,so400m,g}, and the extras tier.  No screen result is used in this order yet.")
A("")
A("## 8. Open items"); A("")
A("- A30, A100 and H100 extraction rates were NOT measured: CLUSTER-A A30 timing job 24842563 (gpuidle) was still pending with an estimated start of 2026-11-07 11:17 local; CLUSTER-B timing jobs 3699969 (A100-RP) and 3699970 (H100-RP) were pending without a start estimate after ~40 min, and all three were cancelled so they would not take queue priority from the backbone arms.  Scripts to rerun: exp/FMD/xs/code/timing_a30.sbatch and exp/FMD/xs/code/clusterB_time_{a100rp,h100rp}.sbatch (CLUSTER-B copy at $U_B/exp/FMD/xs/code/).  Section 3 therefore rests on the A4000 only (the slowest card measured); A100/H100 are not extrapolated here.")
A("- Blocked weights: SD3-medium, FLUX.1-dev, FLUX.1-schnell, SAM3 (HF licence not accepted by the account behind the CLUSTER-A token); InkFM unreleased; C-RADIOv4-{B,L,g} do not exist.")
open(X + "/admin/_capacity_part2.md", "w").write("\n".join(L)); print("ok", choice)
