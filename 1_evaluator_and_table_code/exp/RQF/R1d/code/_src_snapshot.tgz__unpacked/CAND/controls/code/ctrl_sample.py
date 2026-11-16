"""CAND controls: frame_sample.py (frame_code_a44e78b0, CODE_TAG 5dd59e88) unchanged, with the Route A set names that
frame_sample.SETS lacks added (paths and script tags copied from exp/D3_ROUTE/stageA/code/d3a/lane_sample.py SETS,
sha256 73c0e9f1...).  Identity frame only is intended for these sets.  Arguments = frame_sample.py arguments."""
import frame_sample as FS
EXTRA = {
    "test_casia_zh": ("exp/X4a/data/x4a_casia_zh.npz", "casia_zh", "all"),
    "test_casia_en": ("exp/X4a/data/x4a_casia_en.npz", "casia_en", "all"),
    "test_hpl_tamil": ("exp/X4a/data/x4a_hpl_tamil.npz", "hpl_tamil", "all"),
    "german_onhw_all": ("data/d2_v2/onhw/e0_all.npz", "onhw", "all"),
}
for k, v in EXTRA.items():
    assert k not in FS.SETS or FS.SETS[k] == v, k
    FS.SETS[k] = v
if __name__ == "__main__":
    FS.main()
