
import shutil
p="$U/exp/FMD/xs/code/xs_probe_add.py"
shutil.copy(p,p+".bak_before_sat7b")
s=open(p).read()
s=s.replace('("dinov3-vitl16-sat493m", {}),','("dinov3-vitl16-sat493m", {}), ("dinov3-vit7b16-sat493m", dict(dtype=torch.bfloat16)),',1)
open(p,"w").write(s); print(s.count("vit7b16-sat"))
