
p="$U/exp/FMD/xs/code/build_manifest_v2.py"
s=open(p).read()
# 1 generic name mapping
s=s.replace("def lt_for(name):\n    names = LTMAP.get(name, [name]); out = []",
"LTMAP.update({\"qari-ocr-0.4.0-vl-4b\": [\"qari-ocr-0.4.0-vl-4b-vision\"], \"pe-core-g14-448-timm\": [\"pe-core-g14-448\"], \"pe-lang-l14-448-timm\": [\"pe-lang-l14-448\"], \"pe-lang-g14-448-timm\": [\"pe-lang-g14-448\"]})\ndef lt_for(name):\n    names = LTMAP.get(name, [name]); out = []")
# 2 addendum flag
s=s.replace("        m[\"files\"] = j[\"files\"]; m[\"load_test\"] = lt_for(n)","        m[\"addendum1\"] = bool(str(j.get(\"group\", \"\")).startswith(\"add_\") or j.get(\"group\") == \"gated\")\n        m[\"files\"] = j[\"files\"]; m[\"load_test\"] = lt_for(n)")
open(p,"w").write(s)
print("lt_for" in s, s.count("addendum1"))
