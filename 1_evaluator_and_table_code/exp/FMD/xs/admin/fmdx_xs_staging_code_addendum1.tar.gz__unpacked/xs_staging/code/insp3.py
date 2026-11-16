
import numpy as np, glob
for f in sorted(glob.glob("$U/exp/FMD/xs/data/xs_render224_*.npz")):
    Z=np.load(f); print(f.split("/")[-1], "writer_ids uniq",len(np.unique(Z["writer_ids"])),"sample_ids uniq",len(np.unique(Z["sample_ids"])), Z["sample_ids"][:3], Z["writer_ids"][:3], Z["img224"].shape)
