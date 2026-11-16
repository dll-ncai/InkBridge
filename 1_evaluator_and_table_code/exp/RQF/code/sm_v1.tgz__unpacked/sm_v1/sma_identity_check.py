"""Smoke: (1) SMAUnet1D with gate 0 == backbone forward exactly; (2) at init (B = 0) gate 1 == backbone; (3) after perturbing B, gate 1 != backbone.
Usage: sma_identity_check.py KIND BACKBONE_CKPT"""
import sys, torch
sys.path.insert(0, __import__("os").path.dirname(__file__))
import sma_model as SM
kind, path = sys.argv[1], sys.argv[2]
unet, info = SM.load_backbone(kind, path, "ema")
hw = 64 if kind == "luo" else 224
torch.manual_seed(0); x = torch.randn(2, 4, 320); t = torch.tensor([10, 500]); c = torch.rand(2, 1, hw, hw)
with torch.no_grad():
    ref = unet.eval()(x, t, c)
moe = SM.make_moe(kind, unet).eval()
def fwd(g):
    orig = SM.MoEUnet1D.gate_from_condition
    fm = moe.backbone.encoder(c)
    moe.gate_from_condition = lambda cls: (fm, torch.tensor(g, dtype=torch.float32).view(-1, 1))
    with torch.no_grad(): return moe(x, t, c)
d0 = (fwd([0., 0.]) - ref).abs().max().item(); d1 = (fwd([1., 1.]) - ref).abs().max().item()
with torch.no_grad():
    for w in moe._wrappers: w.B[0].normal_(0, 0.02)
d2g1 = (fwd([1., 0.])[0] - ref[0]).abs().max().item(); d2g0 = (fwd([1., 0.])[1] - ref[1]).abs().max().item()
print(dict(kind=kind, n_wrapped=len(moe._wrappers), adapter_params=moe.count_params()["adapters_per_expert"], gate0_maxdiff=d0, init_gate1_maxdiff=d1,
           perturbed_gate1_maxdiff=d2g1, perturbed_gate0_unit_maxdiff=d2g0))
assert d0 == 0.0 and d1 == 0.0 and d2g1 > 0 and d2g0 == 0.0, "identity check failed"
print("IDENTITY_CHECK_OK")
