
import json, glob, csv, os
X="$U/exp/FMD/xs/admin"
with open(X+"/timing_summary.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["gpu_tag","gpu","fm","units_per_s","batch_size","peak_gpu_mem_gb","dtype","token_map_hw_at_224","channels","input_mode","autocast","xs_probe_sha12","job"])
    for p in sorted(glob.glob(X+"/timing/*.json")):
        j=json.load(open(p)); i=j.get("info",{})
        w.writerow([j["tag"],j["env"].get("gpu"),j["name"],j.get("units_per_s"),j.get("batch_size"),j.get("peak_gpu_mem_gb"),i.get("dtype"),json.dumps(j.get("token_map_hw_at_224")),json.dumps(j.get("channels")),i.get("input_mode_(px,interpolate_pos_encoding)",""),j.get("autocast"),j["env"].get("code_sha256","")[:12],j["env"].get("slurm_job")])
with open(X+"/loadtest_summary.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["test","status","token_map_hw_at_224","dtype","input_px","device","xs_probe_sha12","note_or_error"])
    for p in sorted(glob.glob(X+"/loadtest/*.json")):
        j=json.load(open(p)); i=j.get("info",{})
        st=j["status"]; err=(j.get("error_type","")+": "+j.get("error",""))[:300].replace("\n"," ") if st!="ok" else (i.get("convention") or i.get("note") or "")
        if st!="ok" and ("instantiate-only" in j.get("error","") or "loaded (class" in j.get("error","")): st="loaded_no_extractor"
        w.writerow([j["name"],st,json.dumps(j.get("token_map_hw_at_224")),i.get("dtype"),i.get("input_px"),j["env"]["device"],j["env"].get("code_sha256","")[:12],err])
print("ok")
