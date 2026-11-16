#!/usr/bin/env python3
"""RQF R2d sampler for 64-px Trainer1D models (LTO, LTO-FT): exp/CAND/code/frame_code_a44e78b0/frame_sample.py run UNCHANGED
with extra entries added to its SETS table (the six non-Urdu test sets not listed there + the R2d Tamil val split). Same CLI."""
import frame_sample as FS
FS.SETS.update({
    "test_hpl_tamil": ("exp/X4a/data/x4a_hpl_tamil.npz", "hpl_tamil", "all"),
    "test_casia_zh": ("exp/X4a/data/x4a_casia_zh.npz", "casia_zh", "all"),
    "test_casia_en": ("exp/X4a/data/x4a_casia_en.npz", "casia_en", "all"),
    "german_onhw_all": ("data/d2_v2/onhw/e0_all.npz", "onhw", "all"),
    "val1000_tamil": ("exp/RQF/R2d/data/tamil_val1000.npz", "hpl_tamil", "all"),
    "smoke_val_ouhdl": ("data/m5_multiscript_v1/m5_val_ouhdl.npz", "ouhdl", "all"),
})
if __name__ == "__main__":
    FS.main()
