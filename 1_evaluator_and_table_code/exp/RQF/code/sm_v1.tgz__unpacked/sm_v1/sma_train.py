"""SM-A training = D2 d2moe_train.py (unchanged file, imported) with three patches:
(1) build_moe -> sma_model.make_moe (n_experts forced to 1; backbone kind luo | bb),
(2) for kind bb the training dataset is the 224-px render path (sma_model.RenderDataset; --render-npy),
(3) CODE_FILES adds the sm_v1 files so run_config/checkpoints carry their sha256.
Extra args (removed before d2moe_train parses): --backbone-kind {luo,bb}  --render-npy PATH (bb only).
Recipe: d2moe_train defaults/flags given by the job (rank 16, alpha 16, cross,attn,linear, lb-coef 0, lr 1e-4,
batch 64 x accum 4, 10,000 steps, save every 1,000, EMA 0.995/10)."""
import argparse
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sma_model as SM  # noqa: E402
import d2moe_train as DT  # noqa: E402

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--backbone-kind", choices=("luo", "bb"), required=True)
pre.add_argument("--render-npy", type=Path)
A, rest = pre.parse_known_args()
sys.argv = [sys.argv[0]] + rest
if "--n-experts" in rest and rest[rest.index("--n-experts") + 1] != "1":
    raise SystemExit("SM-A uses exactly one adapter (--n-experts 1)")


def build_moe(a, dev):
    a.n_experts = 1
    unet, info = SM.load_backbone(A.backbone_kind, a.backbone, a.backbone_weights)
    moe = SM.make_moe(A.backbone_kind, unet, a.rank, a.lora_alpha, a.lora_targets.split(","), a.gate_hidden, a.gate_feat)
    return moe.to(dev), info


DT.build_moe = build_moe
DT.CODE_FILES = DT.CODE_FILES + [HERE / "sma_model.py", HERE / "sma_train.py", HERE / "sma_sample.py"]
if A.backbone_kind == "bb":
    if A.render_npy is None:
        raise SystemExit("--render-npy required for --backbone-kind bb")
    _orig = DT.U.MultiScriptDataset

    def _ds(npz_paths, condition="rendered", scripts_keep=None, script_override=None, max_units_per_script=None, seed=0):
        p = npz_paths[0] if isinstance(npz_paths, (list, tuple)) else npz_paths
        return SM.RenderDataset(p, A.render_npy, script_override or "ouhdl")
    DT.U.MultiScriptDataset = _ds
if __name__ == "__main__":
    DT.main()
