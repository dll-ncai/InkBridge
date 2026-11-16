
import numpy as np, collections
Z=np.load("$U/data/m5_multiscript_v1/m5_luo_train.npz",allow_pickle=False)
s=Z["scripts"]; print(collections.Counter(s.tolist()))
for k in sorted(set(s.tolist())):
    m=s==k; print(k, int(m.sum()), "writers", len(np.unique(Z["sample_ids"][m])))
