
import json,glob
for f in sorted(glob.glob("$U/exp/FMD/xs/weights/*.manifest_part.json")):
    j=json.load(open(f)); print(j["local_name"], j["status"], j.get("total_bytes"), j.get("error_type"))
