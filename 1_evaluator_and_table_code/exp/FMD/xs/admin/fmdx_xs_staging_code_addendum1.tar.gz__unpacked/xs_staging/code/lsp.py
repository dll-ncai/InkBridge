import json,glob
for f in sorted(glob.glob("$U/exp/FMD/xs/weights/*.manifest_part.json")):
    j=json.load(open(f))
    if str(j.get("group","")).startswith("add_") or j.get("group")=="gated": print(j["local_name"],"|",j["repo_id"],"|",j.get("revision","")[:8],"|",j.get("license"),"|",j.get("gated"))
