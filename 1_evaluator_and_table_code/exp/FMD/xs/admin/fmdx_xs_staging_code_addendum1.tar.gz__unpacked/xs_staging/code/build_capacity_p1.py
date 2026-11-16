#!/usr/bin/env python3
"""Build exp/FMD/xs/admin/CAPACITY_NOTE.md from admin/timing/*.json (+ CLUSTER-B timing copied to admin/timing_clusterB/), loadtest, live_state, luo probe.  Track XS staging (<agent>).  Facts only."""
import json, glob, os, sys, datetime, collections
X = "$U/exp/FMD/xs"
T = collections.defaultdict(dict)    # fm -> gpu tag -> rec
for d in (X + "/admin/timing", X + "/admin/timing_clusterB"):
    for f in sorted(glob.glob(d + "/*.json")):
        j = json.load(open(f))
        if j.get("status") == "ok": T[j["name"]][j["tag"]] = j
tags = [t for t in ("a4000", "a30", "a100rp", "h100rp") if any(t in v for v in T.values())]
gname = {t: next((r["env"].get("gpu") for v in T.values() for k, r in v.items() if k == t), t) for t in tags}
LT = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(X + "/admin/loadtest/*.json")}
LS = json.load(open(X + "/admin/live_state_2026-11-03.json")); PROBE = json.load(open(X + "/admin/luo_render_probe.json"))
N_LUO = 80000
SET_PRI = ["dinov3-vitl16", "cleandift-sd21", "sd21-dift-gate", "pe-spatial-l14-448", "c-radiov4-h"]
SET_ALL = ["dinov3-vitl16", "sd21-dift-gate", "cleandift-sd21", "cleandift-sd15", "pe-spatial-b16-512", "pe-spatial-l14-448", "pe-spatial-g14-448", "c-radiov3-l", "c-radiov3-h", "c-radiov3-g", "c-radiov4-h", "c-radiov4-so400m",
           "siglip2-giant-opt-patch16-256", "dinov3-vith16plus", "dinov3-vit7b16", "tipsv2-so400m14", "tipsv2-g14", "dinov3-convnext-large", "siglip2-so400m-patch14-224", "dinov2-large"]
def rate(fm, tag): return T.get(fm, {}).get(tag, {}).get("units_per_s")
def gpuh(fm, tag, n): 
    r = rate(fm, tag); return None if not r else n / r / 3600
def fmt(x, nd=2): return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))
L = []; A = L.append
A("# CAPACITY NOTE - FMDX staging (track XS staging, <agent>)"); A("")
A(f"Built {datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%MZ')}.  Facts only.  Sources: admin/timing/*.json (CLUSTER-A), admin/timing_clusterB/*.json (CLUSTER-B), admin/loadtest/*.json, admin/live_state_2026-11-03.json, admin/luo_render_probe.json, data/m5_multiscript_v1/DATA_CARD.md."); A("")
A("## 1. Training data of the LUO backbone (units per source)"); A("")
A("LUO backbone = `exp/M5/runs/luo` trained on `data/m5_multiscript_v1/m5_luo_train.npz` (sha256 690441b6bd7c119d774066622c53f23a0aca6e62825a04014351bfba98260b67; 80,000 units; Urdu left out; 100k steps; effective batch 512 sequences/step per prereg A6).  Counts read from the npz (`scripts`, `sample_ids`):"); A("")
A("| source (script tag) | units | writers (sample_ids) | train pool units | units with step > 1 px (val250 rate) |"); A("|---|---|---|---|---|")
A("| CASIA-OLHWDB zh (casia_zh) | 20,000 | 552 | 2,105,324 | 46.8% (val) |"); A("| CASIA-OLHWDB en (casia_en) | 20,000 | 552 | 28,699 | 0.0% (val) |")
A("| HPL Tamil (hpl_tamil) | 20,000 | 111 | 49,156 | 2.0% (val) |"); A("| IAM-OnDB components (iam) | 20,000 | 382 | 81,659 | 0.0% (val) |"); A("| **total LUO** | **80,000** | | | |"); A("")
A("Not in the LUO mix: OUHD-L Urdu (20,000 sampled in the M5 joint set; p01 FT subset = 336 units / 183 writers, p10 = 3,362, p30 = 10,085, p100 = 33,615), German UNIPEN-de and OnHW (zero-shot).  Screen sets rendered at 224 px (this track): casia_zh / casia_en / hpl_tamil / iam val250 and the UNIPEN-de train stand-in (3,927 units, 12 writers); 64-px re-render byte-identical for every unit.  Files: `exp/FMD/xs/data/xs_render224_*.npz` (+ `.meta.json`).")
A("")
A("**Training-render fact (LUO units carry no ground-truth ink):** `m5_luo_train.npz` has `seq` (Nagamatsu) and `cond_rendered` (64 px) but no `gt_points`.  224-px renders therefore have to be derived by `e0_adapter.decode(seq)` -> `fmd_harness.render.render224`.  Probe (admin/luo_render_probe.json, 125 random units per script, seed 20261103):"); A("")
A("| script | units probed | decoded-seq 64-px render byte-identical to stored cond_rendered | max abs diff | mean abs diff (grey levels) |"); A("|---|---|---|---|---|")
for s in ("casia_zh", "casia_en", "hpl_tamil", "iam"):
    p = PROBE[s]; A(f"| {s} | {p['n']} | {p['identical_64']} | {p['max_abs_diff']} | {p['mean_abs_diff']} |")
sp = PROBE["speed_s_per_unit"]; A(""); A(f"Speed (1 CPU core): decode {sp['decode']*1e3:.2f} ms, render64 {sp['render64']*1e3:.2f} ms, render224 {sp['render224']*1e3:.2f} ms per unit -> 80,000 units ~ {80000*(sp['decode']+sp['render224'])/60:.1f} CPU-min.  RTL/mirror_x framing of training renders is not included in this figure."); A("")
A("## 2. Per-FM extraction rate (1,000 val1000 renders, strict fp32, TF32 off, forward + copy of returned layers to CPU, weight load excluded)"); A("")
hdr = "| FM | token map at 224 px | channels | " + " | ".join(f"{t} u/s" for t in tags) + " | code sha (first run) |"; A(hdr); A("|---|---|---|" + "---|"*len(tags) + "---|")
order = [f for f in SET_ALL if f in T] + sorted(f for f in T if f not in SET_ALL)
for fm in order:
    anyr = next(iter(T[fm].values())); hw = anyr.get("token_map_hw_at_224") or (LT.get(fm, {}).get("token_map_hw_at_224")); ch = anyr.get("channels")
    A(f"| {fm} | {json.dumps(hw) if hw else 'see loadtest'} | {json.dumps(ch) if ch else ''} | " + " | ".join(fmt(rate(fm, t), 1) for t in tags) + f" | {anyr['env'].get('code_sha256','')[:8]} |")
A(""); A("GPUs: " + "; ".join(f"{t} = {gname[t]}" for t in tags) + ".  `bf16`: dinov3-vit7b16 is loaded in bf16 (27 GB fp32 does not fit a 16/24 GB card); every other row is fp32.  sd21-dift-gate = DIFT gate layer only (768 px, 4 noise draws x 1 timestep, UNet stopped after up_block 1); cleandift-sd21 = one clean-latent pass at t = 0 through the full UNet at 768 px (load-test convention, NOT the official `get_features` path).")
A("")
A("## 3. GPU-hours for REPA targets"); A("")
A("GPU-h = units / (units/s) / 3600 on one GPU of that type, one pass over the units, extraction only (no probe/training time).  Rows use the measured rate of the FM on that GPU type.")
A("")
for nm, S in (("priority set (control has no FM; DINOv3-L, CleanDIFT-SD2.1, SD2.1-DIFT gate, PE-Spatial-L14 as PE proxy, C-RADIOv4-H)", SET_PRI), ("all roster candidates with a measured rate", SET_ALL)):
    A(f"### {nm}"); A("")
    A("| FM | " + " | ".join(f"{t}: 80k units GPU-h" for t in tags) + " |"); A("|---|" + "---|"*len(tags))
    tot = {t: 0.0 for t in tags}; miss = {t: [] for t in tags}
    for fm in S:
        row = []
        for t in tags:
            h = gpuh(fm, t, N_LUO)
            if h is None: miss[t].append(fm)
            else: tot[t] += h
            row.append(fmt(h))
        A(f"| {fm} | " + " | ".join(row) + " |")
    A("| **sum over rows with a rate** | " + " | ".join(f"**{tot[t]:.2f}**" for t in tags) + " |"); 
    A("| rows without a rate | " + " | ".join(str(len(miss[t])) for t in tags) + " |"); A("")
    A(f"Subset size that keeps the sum at <= 24 GPU-h (per tag, linear scaling): " + "; ".join(f"{t}: {min(N_LUO, int(24 / tot[t] * N_LUO)) if tot[t] > 0 else 'n/a'} units" for t in tags)); A("")
open(X + "/admin/_capacity_part1.md", "w").write("\n".join(L))
print("\n".join(L)[:200]); print("tags", tags, "fms", len(T))
