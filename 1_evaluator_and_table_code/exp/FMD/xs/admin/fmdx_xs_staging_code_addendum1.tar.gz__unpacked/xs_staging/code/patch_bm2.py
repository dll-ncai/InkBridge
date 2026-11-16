
p="$U/exp/FMD/xs/code/build_manifest_v2.py"
s=open(p).read()
s=s.replace('L.append(f"| {m[\'local_name\']} | {m[\'repo_id\']}','L.append(f"| {m[\'local_name\']}{\' [A1]\' if m.get(\'addendum1\') else \'\'} | {m[\'repo_id\']}',1)
s=s.replace('"## Staged", "",','"## Staged", "", "[A1] = staged in addendum 1 (lead follow-up 2026-11-03: stage every discovery-list candidate; initial MANIFEST_v2 preserved as MANIFEST_v2_initial.json/.md).", "",',1)
open(p,"w").write(s); print(s.count("[A1]"))
import json; e=json.load(open("$U/exp/FMD/xs/admin/manifest_v2_extra.json"))
print(list(e)); print(json.dumps(e)[:3500])
