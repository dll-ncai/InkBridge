
import numpy as np, glob, json
for f in sorted(glob.glob("$U/data/m5_multiscript_v1/m5_val_*.npz"))+["$U/data/m5_multiscript_v1/m5_luo_train.npz"]:
    Z=np.load(f,allow_pickle=True)
    print(f.split("/")[-1], {k:(Z[k].shape,str(Z[k].dtype)) for k in Z.files})
