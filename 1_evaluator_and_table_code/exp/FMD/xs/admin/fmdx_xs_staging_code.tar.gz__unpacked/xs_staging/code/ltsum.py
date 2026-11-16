
import json,glob,os
for f in sorted(glob.glob("$U/exp/FMD/xs/admin/loadtest/*.json")):
    j=json.load(open(f)); 
    print(j["name"], j["status"], j.get("token_map_hw_at_224") or (j.get("error_type")+": "+j.get("error","")[:140].replace("\\n"," ")), j.get("info",{}).get("input_mode_(px,interpolate_pos_encoding)",""), j["env"]["date_utc"])
