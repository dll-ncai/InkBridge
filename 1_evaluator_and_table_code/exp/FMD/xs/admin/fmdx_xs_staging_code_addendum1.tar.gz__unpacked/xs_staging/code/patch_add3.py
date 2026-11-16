
p="$U/exp/FMD/xs/code/xs_probe_add.py"
s=open(p).read()
s=s.replace('"paddleocr-vl-1.0", "florence-2-base", "florence-2-large-ft", "arabic-ocr','"paddleocr-vl-1.0", "arabic-ocr',1)
open(p,"w").write(s); print(s.count('"florence-2-base"'))
