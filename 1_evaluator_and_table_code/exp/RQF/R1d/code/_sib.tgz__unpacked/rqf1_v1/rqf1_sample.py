"""RQF R1: frame_sample.py (frame_code_a44e78b0, CODE_TAG 5dd59e88) unchanged, with extra set names: Route A test sets and the
mirrored-LTR sets built by build_mirrored.py (exp/RQF/R1/data/rqf_mir_*.npz). Arguments = frame_sample.py arguments."""
import frame_sample as FS
EXTRA = {
    "test_casia_zh": ("exp/X4a/data/x4a_casia_zh.npz", "casia_zh", "all"),
    "test_casia_en": ("exp/X4a/data/x4a_casia_en.npz", "casia_en", "all"),
    "test_hpl_tamil": ("exp/X4a/data/x4a_hpl_tamil.npz", "hpl_tamil", "all"),
    "german_onhw_all": ("data/d2_v2/onhw/e0_all.npz", "onhw", "all"),
    "test_iam500_mir": ("exp/RQF/R1/data/rqf_mir_iam500.npz", "iam", "all"),
    "test_casia_en_mir": ("exp/RQF/R1/data/rqf_mir_casia_en.npz", "casia_en", "all"),
    "test_hpl_tamil_mir": ("exp/RQF/R1/data/rqf_mir_hpl_tamil.npz", "hpl_tamil", "all"),
}
for k, v in EXTRA.items():
    assert k not in FS.SETS or FS.SETS[k] == v, k
    FS.SETS[k] = v
if __name__ == "__main__":
    FS.main()
