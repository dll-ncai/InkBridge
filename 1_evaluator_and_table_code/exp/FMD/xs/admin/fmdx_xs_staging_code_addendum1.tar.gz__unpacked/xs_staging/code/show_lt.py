import json
for n in ["arabic-ocr-deepseek-ocr-2"]:
    j=json.load(open("$U/exp/FMD/xs/admin/loadtest/"+n+".json")); print("##",n,j["status"],j.get("token_map_hw_at_224"),json.dumps(j.get("info",{}))[:420]); print("  ",(j.get("error") or "")[:900].replace("\n"," ")); print("  ",(j.get("tb_tail") or "")[-600:].replace("\n"," | "))
