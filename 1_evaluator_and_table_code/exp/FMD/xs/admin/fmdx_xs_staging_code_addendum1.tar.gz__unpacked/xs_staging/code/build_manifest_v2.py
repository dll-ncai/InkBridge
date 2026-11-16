#!/usr/bin/env python3
"""Build exp/FMD/xs/weights/MANIFEST_v2.json/.md from *.manifest_part.json + admin/loadtest/*.json + github clone commits.  Track XS staging."""
import json, glob, os, subprocess, datetime, hashlib
X = "$U/exp/FMD/xs"; W = X + "/weights"
parts = {}
for f in sorted(glob.glob(W + "/*.manifest_part.json")):
    j = json.load(open(f)); parts[j["local_name"]] = j
LT = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(X + "/admin/loadtest/*.json")}
# map local dir name -> load-test name(s)
TIM = {}
for _d in (X + "/admin/timing", X + "/admin/timing_clusterB"):
    for _f in glob.glob(_d + "/*.json"):
        _j = json.load(open(_f))
        if _j.get("status") == "ok": TIM.setdefault(_j["name"], {})[_j["tag"]] = _j["units_per_s"]
LTMAP = {"cleandift": ["cleandift-sd21", "cleandift-sd15"], "sd1.5": ["cleandift-sd15"], "qaari-0.1-urdu-ocr": ["qaari-urdu-vision"], "pe-core-b16-224-timm": ["pe-core-b16-224"], "pe-core-l14-336-timm": ["pe-core-l14-336"], "pe-spatial-g14-448-timm": ["pe-spatial-g14-448"], "pe-spatial-l14-448-timm": ["pe-spatial-l14-448"], "pe-spatial-b16-512-timm": ["pe-spatial-b16-512"],
         "pe-spatial-s16-512-timm": ["pe-spatial-s16-512"], "pe-spatial-t16-512-timm": ["pe-spatial-t16-512"]}
LTMAP.update({"qari-ocr-0.4.0-vl-4b": ["qari-ocr-0.4.0-vl-4b-vision"], "pe-core-g14-448-timm": ["pe-core-g14-448"], "pe-lang-l14-448-timm": ["pe-lang-l14-448"], "pe-lang-g14-448-timm": ["pe-lang-g14-448"]})
def lt_for(name):
    names = LTMAP.get(name, [name]); out = []
    for n in names:
        if n in LT:
            r = LT[n]
            out.append(dict(test=n, status=("loaded_no_extractor" if (r["status"] != "ok" and ("instantiate-only" in r.get("error", "") or "loaded (class" in r.get("error", ""))) else r["status"]), token_map_hw_at_224=r.get("token_map_hw_at_224"), shapes_N4=r.get("shapes_N4"), error=(r.get("error_type", "") + ": " + r.get("error", "")[:300]) if r["status"] != "ok" else None,
                            device=r["env"]["device"], dtype=r.get("info", {}).get("dtype"), input_px=r.get("info", {}).get("input_px") or {"sam3": 1008}.get(n), note=r.get("info", {}).get("convention") or r.get("info", {}).get("note")))
    return out
gh = {}
for d in sorted(glob.glob(W + "/github/*")):
    if os.path.isdir(d + "/.git"):
        try:
            sha = subprocess.check_output(["git", "-C", d, "rev-parse", "HEAD"], text=True).strip(); date = subprocess.check_output(["git", "-C", d, "log", "-1", "--format=%cI"], text=True).strip()
            url = subprocess.check_output(["git", "-C", d, "config", "--get", "remote.origin.url"], text=True).strip()
            gh[os.path.basename(d)] = dict(url=url, commit=sha, commit_date=date, path=d)
        except Exception as e: gh[os.path.basename(d)] = dict(error=str(e))
models = []; blocked = []
for n, j in parts.items():
    if j["status"] == "ok":
        m = {k: j.get(k) for k in ("local_name", "repo_id", "revision", "license", "license_name", "license_link", "gated", "base_model", "role", "group", "total_bytes", "n_files", "safetensors_total_params", "download_date", "command", "allow_patterns", "last_modified")}
        m["addendum1"] = bool(str(j.get("group", "")).startswith("add_") or j.get("group") == "gated")
        m["files"] = j["files"]; m["load_test"] = lt_for(n)
        m["units_per_s"] = {t: v for tn in LTMAP.get(n, [n]) for t, v in TIM.get(tn, {}).items()} or None
        if n.startswith("urdu-ocr-deepseek"): m["note"] = "LoRA adapter only (LM layers only: no vision-encoder tensors); vision encoder = base DeepSeek-OCR / OCR-2; no standalone load test"
        models.append(m)
    else:
        blocked.append(dict(local_name=n, repo_id=j["repo_id"], gated=j.get("gated"), license=j.get("license"), license_name=j.get("license_name"), role=j.get("role"), error_type=j.get("error_type"), error=j.get("error"), revision_seen=j.get("revision")))
V1 = json.load(open("$U/exp/FMD/weights/MANIFEST.json"))
man = dict(schema="fmdx_manifest/2", created_utc=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"), track="XS staging", model="<agent>",
           reason="staging/pulls with sha checks and re-running existing recipes = <agent> (lead model policy 30 Oct 2026)", weights_root=W,
           v1_manifest="$U/exp/FMD/weights/MANIFEST.json (8 FMs, unchanged; sha256 " + hashlib.sha256(open("$U/exp/FMD/weights/MANIFEST.json", "rb").read()).hexdigest() + ")",
           python_env="exp/FMD/venv + exp/FMD/xs/pylibs (timm 1.0.30, einops, open_clip_torch 3.3.0, hydra-core, omegaconf, jaxtyping; freeze exp/FMD/xs/pylibs_freeze.txt) inside images/torch27.sif",
           models=models, blocked=blocked, github=gh)
extra = "$U/exp/FMD/xs/admin/manifest_v2_extra.json"
if os.path.exists(extra): man.update(json.load(open(extra)))
json.dump(man, open(W + "/MANIFEST_v2.json", "w"), indent=1)
L = ["# MANIFEST_v2 - FMDX staging (track XS staging, <agent>)", "", f"Created {man['created_utc']}. Weights root `{W}` (CLUSTER-A only; never copied off CLUSTER-A except the CLUSTER-B mirror listed in admin/mirror/). v1 manifest: {man['v1_manifest']}.", "",
     "Load test = 4 val1000 renders (224 px) through the loader in `exp/FMD/xs/code/xs_probe.py` (LOAD-TEST conventions, not the pre-registered extractors). `token map` = H x W of the returned map(s) at 224-px input.", "",
     "## Staged", "", "[A1] = staged in addendum 1 (lead follow-up 2026-11-03: stage every discovery-list candidate; initial MANIFEST_v2 preserved as MANIFEST_v2_initial.json/.md).", "", "| local dir | repo id | revision | licence | gated | bytes | files | load test (token map at 224 px) | units/s (A4000 / others) |", "|---|---|---|---|---|---|---|---|---|"]
for m in models:
    lt = "; ".join(f"{t['test']}: " + (json.dumps(t['token_map_hw_at_224']) + (f" input {t['input_px']}px" if t.get("input_px") and t['input_px'] != 224 else "") if t["status"] == "ok" else ("LOADS, no extractor: " if t["status"] == "loaded_no_extractor" else "ERROR ") + (t["error"] or "")[:160]) for t in m["load_test"]) or "not run"
    lic = (m["license"] if isinstance(m["license"], str) else ",".join(m["license"] or [])) + (f" / {m['license_name']}" if m.get("license_name") else "")
    L.append(f"| {m['local_name']}{' [A1]' if m.get('addendum1') else ''} | {m['repo_id']} | `{m['revision']}` | {lic} | {m['gated']} | {m['total_bytes']} | {m['n_files']} | {lt} | {json.dumps(m.get('units_per_s')) if m.get('units_per_s') else ('adapter' if m.get('note') else '')} |")
L += ["", "## Blocked / not staged (exact errors)", "", "| local dir | repo id | gated | licence | error |", "|---|---|---|---|---|"]
for b in blocked:
    L.append(f"| {b['local_name']} | {b['repo_id']} | {b['gated']} | {b['license']} {b.get('license_name') or ''} | {b['error_type']}: {(b['error'] or '')[:600].replace(chr(10),' ')} |")
for b in man.get("not_staged", []):
    L.append(f"| - | {b['repo_id']} | {b.get('gated','')} | {b.get('license','')} | {b['reason']} |")
L += ["", "## Code repositories (cloned, pinned)", "", "| repo | url | commit | commit date |", "|---|---|---|---|"]
for k, v in gh.items(): L.append(f"| {k} | {v.get('url')} | `{v.get('commit')}` | {v.get('commit_date')} |")
if man.get("discovery_list"):
    L += ["", "## Further dense-feature / ink / handwriting FMs found by HF search (for the prereg track; NOT screened here)", "", "| repo | note | staged here? |", "|---|---|---|"]
    for d in man["discovery_list"]: L.append(f"| {d['repo_id']} | {d['note']} | {d['staged']} |")
L += ["", "## Per-file sha256", ""]
for m in models:
    L += [f"### {m['local_name']} ({m['repo_id']} @ `{m['revision']}`)", "", "| file | bytes | sha256 |", "|---|---|---|"] + [f"| {f['path']} | {f['bytes']} | `{f['sha256']}` |" for f in m["files"]] + [""]
open(W + "/MANIFEST_v2.md", "w").write("\n".join(L))
print("models", len(models), "blocked", len(blocked), "github", list(gh))
