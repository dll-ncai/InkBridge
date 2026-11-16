import json,glob,collections
g=collections.defaultdict(list)
for f in sorted(glob.glob("$U/exp/FMD/xs/weights/*.manifest_part.json")):
    j=json.load(open(f)); g[(j.get("group"),j["status"])].append(j["local_name"])
for k,v in g.items(): print(k,len(v),v if str(k[0]).startswith("add") or k[1]!="ok" and 1 else "")
