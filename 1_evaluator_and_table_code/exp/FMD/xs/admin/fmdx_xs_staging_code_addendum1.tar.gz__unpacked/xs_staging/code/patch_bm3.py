
p="$U/exp/FMD/xs/code/build_manifest_v2.py"; s=open(p).read()
a='input_px=r.get("info", {}).get("input_px")'
print(s.count(a))
s=s.replace(a,'input_px=r.get("info", {}).get("input_px") or {"sam3": 1008}.get(n)')
open(p,"w").write(s)
