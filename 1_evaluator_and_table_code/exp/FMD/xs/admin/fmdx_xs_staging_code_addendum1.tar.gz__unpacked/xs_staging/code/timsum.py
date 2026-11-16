
import json,glob
for f in sorted(glob.glob("$U/exp/FMD/xs/admin/timing/*.json")):
    j=json.load(open(f)); print(j["tag"], j["name"], j["status"], j.get("units_per_s") or (j.get("error_type")+": "+j.get("error","")[:100].replace("\n"," ")), j.get("batch_size"), j.get("peak_gpu_mem_gb"), j["env"].get("code_sha256","")[:8], j.get("token_map_hw_at_224"))
