import json,glob
for f in sorted(glob.glob("$U/exp/FMD/xs/admin/timing/*.json"),key=lambda f: __import__("os").path.getmtime(f))[-16:]:
    j=json.load(open(f)); print(j["name"],j["tag"],j["status"],j.get("units_per_s"),j.get("batch"),(j.get("error") or "")[:100])
