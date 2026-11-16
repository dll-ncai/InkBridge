#!/usr/bin/env python3
import json, glob, os, hashlib, subprocess, datetime, tarfile
X = "$U/exp/FMD/xs"; R = "$U"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
# ---- code tar (own files only; the fmdx_backbone_v1 dir belongs to another track)
codefiles = sorted(p for p in glob.glob(X + "/code/*") if os.path.isfile(p) and not os.path.basename(p).startswith("slurm-"))
extra = [X + "/pylibs_freeze.txt", X + "/pylibs_ds_freeze.txt"] + [p for p in [X + "/pylibs_clusterB_freeze.txt"] if os.path.exists(p)]
tarp = X + "/admin/fmdx_xs_staging_code.tar.gz"
with tarfile.open(tarp, "w:gz") as t:
    for p in codefiles + extra: t.add(p, arcname="xs_staging/" + os.path.relpath(p, X))
code_sha = {os.path.basename(p): sha(p) for p in codefiles}
probe_versions = sorted({json.load(open(f))["env"].get("code_sha256", "")[:12] for f in glob.glob(X + "/admin/timing/*.json")})
def fsha(rel): 
    p = X + "/" + rel; return dict(path=p, sha256=sha(p), bytes=os.path.getsize(p))
outs = {k: fsha(k) for k in ["weights/MANIFEST_v2.json", "weights/MANIFEST_v2.md", "admin/CAPACITY_NOTE.md", "admin/live_state_2026-11-03.json", "admin/luo_render_probe.json", "admin/hf_search_raw.txt",
        "admin/clusterB_expected.json", "admin/manifest_v2_extra.json", "data/luo_repa_order_v1.csv", "data/luo_repa_order_v1.csv.meta.json", "data/xs_render224_summary.json"]}
outs["code_tar"] = dict(path=tarp, sha256=sha(tarp), bytes=os.path.getsize(tarp), n_files=len(codefiles) + len(extra))
renders = {}
for f in sorted(glob.glob(X + "/data/xs_render224_*.npz")): renders[os.path.basename(f)] = dict(sha256=sha(f), meta_sha256=sha(f + ".meta.json"))
mir = {}
for f in sorted(glob.glob(X + "/admin/mirror/mirror_*.sha256")):
    tag = os.path.basename(f)[7:-7]; mir[tag] = dict(list_sha256=sha(f[:-7] + ".list"), sha256_listing_sha256=sha(f))
verify = {}
for f in glob.glob(X + "/admin/mirror/clusterB_verify_*.txt"):
    ls = open(f).read().strip().splitlines(); verify[os.path.basename(f)] = dict(lines=len(ls), ok=sum(l.endswith(": OK") for l in ls), note="sha256sum -c at CLUSTER-B login1 against the CLUSTER-A-created listing (CLUSTER-A -> VPS hub -> CLUSTER-B)")
verify["vps_verify_A_B_24842665.txt"] = dict(summary=open(X + "/admin/mirror/vps_verify_A_B_24842665.txt").read().strip().replace("\r", "").splitlines()[-2:], note="Get-FileHash on the VPS: B 38/38 OK; A 6,901/6,914 OK, the 13 BAD files were edited on CLUSTER-A after the listing (12 belong to another track's code dir under xs/code, plus xs_probe.py)")
verify["clusterB_hf_direct_download"] = "admin/clusterB_stage_result.json: 8/9 repos all files sha-OK vs CLUSTER-A manifest; sd2.1 14/14 expected files sha-OK (14 extra unrequested variant files downloaded, then deleted)"
parts = {os.path.basename(f)[:-len(".manifest_part.json")]: sha(f) for f in sorted(glob.glob(X + "/weights/*.manifest_part.json"))}
man = json.load(open(X + "/weights/MANIFEST_v2.json"))
prov = dict(schema="fmdx_provenance/1", track="XS staging (FMDX phase 0)", created_utc=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
 sub_agent_model="<agent>", model_reason="staging/pulls with sha checks and re-running existing render/sizing recipes = <agent> per the lead's model policy (30 Oct 2026)",
 prereg=dict(path=R + "/exp/FMD/plans/FMD_PREREG_2026-11-03.md", sha256_at_build=sha(R + "/exp/FMD/plans/FMD_PREREG_2026-11-03.md"), lead_message="A6 appended ~13:32Z; sha 3c1beaa8... ; A3 screen rules unchanged since d5137178"),
 staged=dict(n_models_ok=len(man["models"]), n_blocked=len(man["blocked"]), blocked=[b["local_name"] + ": " + b["error_type"] for b in man["blocked"]], manifest_part_sha256=parts),
 outputs=outs, renders=renders, mirror_lists=mir, clusterB_verify_last_lines=verify,
 inputs=dict(m5_val_npz_dir=R + "/data/m5_multiscript_v1", m5_luo_train_npz_sha256="690441b6bd7c119d774066622c53f23a0aca6e62825a04014351bfba98260b67 (DATA_CARD)", unipen_de_train=R + "/data/d2_v2/unipen_de/e0_train.npz", harness=R + "/exp/FMD/code/fmd_harness_v1 (fmd_harness 1.0.1)", val1000_render=sha(R + "/exp/FMD/data/fmd_render224_val1000.npz")),
 code_sha256=code_sha, xs_probe_versions_in_timing=probe_versions,
 jobs=dict(
  clusterA_staging=[24841106, 24841107, 24841108, 24841109, 24841110, 24841111, 24841114, 24842025, 24842361], clusterA_git=[24841128, 24841674], clusterA_pkgs=[24841619, 24841677, 24842323],
  clusterA_render=dict(run1=24841130, final=24841603), clusterA_loadtests_completed=[24841665, 24841670, 24841671, 24841672, 24842336, 24842339, 24842341, 24842342, 24842364, 24842365, 24842366, 24842368, 24842370, 24842374, 24842379, 24842564, 24842565, 24842658],
  clusterA_loadtests_cancelled=[24841666, 24841668, 24842326], clusterA_timing_completed=[24842562, 24843116, 24843117, 24843187], clusterA_timing_cancelled=[24842474, 24842483, 24842484, 24842477, 24842563],
  clusterA_sizing=dict(luo_render_probe=24842638, luo_render_probe_cancelled_slow_first_version=24842569, repa_order=24843166), clusterA_hub=dict(push=[24841888, 24843208, 24843292], vps_verify=24842665),
  clusterB=dict(failed_pull_job_no_key_on_compute_nodes=3699811, setup_and_hf_download=3699856, timing_cancelled=[3699969, 3699970])),
 deviations=["CLUSTER-B mirror of weights: public weights were downloaded directly from HF at the pinned revision and sha256-verified against the CLUSTER-A manifest (VPS hub blocked 13:35-15:05Z); DINOv3 weights not mirrored (licence)",
  "A30 / A100 / H100 extraction rates not measured (jobs pending, cancelled); only A4000 measured",
  "load tests are CPU/GPU load-test conventions, not the prereg extractors; DeepSeek-OCR runs with transformers 4.46.3 (pylibs_ds), everything else transformers 5.22.1",
  "StarVector image encoder rebuilt from the repo's clip_model.py with fairscale stubbed (not via the starvector package)"])
json.dump(prov, open(X + "/PROVENANCE_FMDX_staging.json", "w"), indent=1)
print("ok", len(json.dumps(prov)), outs["code_tar"])
