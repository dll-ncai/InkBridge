import json
e=json.load(open("$U/exp/FMD/xs/admin/manifest_v2_extra.json"))
for d in e["discovery_list"]: print(d["repo_id"],"|",d["staged"],"|",d["note"][:110])
