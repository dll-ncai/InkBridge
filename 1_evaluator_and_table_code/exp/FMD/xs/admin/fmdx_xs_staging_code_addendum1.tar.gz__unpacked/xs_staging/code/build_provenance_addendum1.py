#!/usr/bin/env python3
"""PROVENANCE_FMDX_staging_addendum1.json (+ code tar of all xs/code files).  Track XS staging."""
import json, glob, os, hashlib, datetime, tarfile
X="$U/exp/FMD/xs"; R="$U"
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<22),b""): h.update(b)
    return h.hexdigest()
codefiles=sorted(p for p in glob.glob(X+"/code/*") if os.path.isfile(p) and not os.path.basename(p).startswith("slurm-"))
extra=[p for p in [X+"/pylibs_freeze.txt",X+"/pylibs_ds_freeze.txt"] if os.path.exists(p)]
tarp=X+"/admin/fmdx_xs_staging_code_addendum1.tar.gz"
with tarfile.open(tarp,"w:gz") as t:
    for p in codefiles+extra: t.add(p,arcname="xs_staging/"+os.path.relpath(p,X))
man=json.load(open(X+"/weights/MANIFEST_v2.json"))
a1=[m for m in man["models"] if m.get("addendum1")]
parts={m["local_name"]:sha(X+"/weights/"+m["local_name"]+".manifest_part.json") for m in a1}
blocked_parts={os.path.basename(f):sha(f) for f in glob.glob(X+"/weights/*blocked_1st_attempt.json")}
lt={}
for m in a1:
    for t in m["load_test"]:
        f=X+"/admin/loadtest/"+t["test"]+".json"
        if os.path.exists(f): lt[t["test"]]=dict(sha256=sha(f),status=t["status"],env=json.load(open(f))["env"].get("code_sha256","")[:12],add_sha=json.load(open(f))["env"].get("add_sha256","")[:12])
for n in ["florence-2-large-ft","florence-2-base","dots-ocr","dots-ocr-base","sam3.1"]:
    f=X+"/admin/loadtest/"+n+".json"
    if os.path.exists(f): lt.setdefault(n,dict(sha256=sha(f)))
tim={}
for f in glob.glob(X+"/admin/timing/*.json"):
    j=json.load(open(f))
    if j.get("status")=="ok" and j["name"] in {t for m in a1 for t in [m["local_name"]]+[x["test"] for x in m["load_test"]]}: tim[j["name"]]=dict(sha256=sha(f),units_per_s=j["units_per_s"],code_sha=j["env"].get("code_sha256","")[:12],add_sha=j["env"].get("add_sha256","")[:12],gpu=j["env"].get("device"))
deit={}
for d in glob.glob(X+"/weights/.hf_home/hub/models--facebook--deit-*"):
    deit[os.path.basename(d)]=sorted(os.listdir(d+"/snapshots"))
outs={k:dict(path=X+"/"+k,sha256=sha(X+"/"+k)) for k in ["weights/MANIFEST_v2.json","weights/MANIFEST_v2.md","weights/MANIFEST_v2_initial.json","weights/MANIFEST_v2_initial.md","admin/manifest_v2_extra.json","admin/manifest_v2_extra_initial.json","admin/ADDENDUM1_SUMMARY.md","admin/vision_identity_addendum1.json","admin/sam3.1_ckpt_keys.json","admin/CAPACITY_NOTE.md"] if os.path.exists(X+"/"+k)}
outs["code_tar"]=dict(path=tarp,sha256=sha(tarp),n_files=len(codefiles)+len(extra))
prov=dict(schema="fmdx_provenance_addendum/1",track="XS staging addendum 1",created_utc=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"),sub_agent_model="<agent>",
 request="Lead follow-up 2026-11-03 (<author>: fetch ALL candidates) + HF licences accepted ~15:20Z (gated re-fetch); same conventions as PROVENANCE_FMDX_staging.json; no extractors / no screen written",
 prereg=dict(path=R+"/exp/FMD/plans/FMD_PREREG_2026-11-03.md",sha256_at_build=sha(R+"/exp/FMD/plans/FMD_PREREG_2026-11-03.md")),
 n_models_staged_addendum1=len(a1),models_addendum1={m["local_name"]:dict(repo_id=m["repo_id"],revision=m["revision"],bytes=m["total_bytes"],n_files=m["n_files"]) for m in a1},
 blocked=[dict(local_name=b["local_name"],repo_id=b["repo_id"],error_type=b["error_type"],error=b["error"]) for b in man["blocked"] if b["local_name"].startswith("sd3.5")],
 manifest_part_sha256=parts,blocked_first_attempt_parts_sha256=blocked_parts,loadtest_json=lt,timing_json=tim,deit_hf_cache_snapshots=deit,outputs=outs,
 code_sha256={os.path.basename(p):sha(p) for p in codefiles},
 jobs=dict(staging=dict(add_named=24844304,add_ocr1=24844305,add_ocr2=24844306,add_ssl1=24844307,add_ssl2=24844308,add_misc=24844309,named2=24844914,ocr1b=24844915,cfg=24844926,add_sat7b=24846110,gated_refetch=dict(sd3_medium=24844920,flux1_dev=24844921,flux1_schnell=24844922,sam3=24844923),prefetch_deit=[24845045,24845274,24845493]),
  loadtests=dict(b1a=24844929,b1b=24844930,b2a=24844950,b3a=24844979,b3b=24844980,b4a=24845062,b5theia=24845206,b5sd3=24845207,b6a=24845236,b6b=24845237,b7flux=24845256,b8theia=24845257,b9a=24845275,b10theia=24845303,b11theia=24845922,b12a=24845927,b12b=24845928,b13sat7b=24846155,b14florence=[24846162,24846163],b15florence=[24846173,24846174],b16florence=24846183,b17arabic_deepseek_ds=24846196),
  timing=dict(add1=24844995,add2_cancelled_in_queue_replaced_by_split=24845277,add3_cancelled_in_queue_replaced_by_split=24846156,split=dict(s1=24846330,s2=24846335,s3=24846340,s4=24846341)),analysis=dict(sam31_keys=24845924,vision_identity=24845930,reverify_5ids=24846250)),
 deviations=["Theia remote code needs transformers 4.46.3 (XS_DS=1 stack) and loads facebook/deit-{tiny,small,base}-patch16-224 inside __init__; those repos (config + pytorch_model.bin) were prefetched into weights/.hf_home (snapshot ids in deit_hf_cache_snapshots)",
  "Florence-2-large-ft load test uses the XS_DS=1 stack (transformers 5.22.1 remote code AttributeError forced_bos_token_id); Florence-2-base on 5.22.1",
  "SD3 / FLUX / sam3 load tests are LOAD-TEST-only conventions (SD3 and FLUX: VAE latent, flow sigma 0.25, empty prompt; FLUX on CPU, N=1); sam3 token map is for its native ~1008 px input",
  "FLUX.1 A4000 timing not measured (bf16 DiT ~24 GB > 16 GB); sam3.1 and arabic-hw-ocr-4bit and PE-Core-S16/T16 not load-tested (reasons in manifest_v2_extra.json)",
  "arabic-ocr-deepseek-ocr-2 not load-tested: remote code needs DeepseekV2Model (ImportError on transformers 4.46.3); its 472 vision/projector tensors are identical to deepseek-ocr-2 (vision_identity_addendum1.json)",
  "SD3.5 medium/large/large-turbo not staged: GatedRepoError 403 (separate Stability AI licence click-through)",
  "DINOv3 satellite weights (vitl16, vit7b16) stay on CLUSTER-A; nothing from addendum 1 mirrored to CLUSTER-B (capacity note unchanged: extraction on CLUSTER-A)"])
json.dump(prov,open(X+"/PROVENANCE_FMDX_staging_addendum1.json","w"),indent=1); print("ok",len(a1),len(lt),len(tim),outs["code_tar"])
