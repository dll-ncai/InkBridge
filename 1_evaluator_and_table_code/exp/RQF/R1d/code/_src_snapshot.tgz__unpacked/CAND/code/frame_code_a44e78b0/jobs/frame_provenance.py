"""Write eval/PROVENANCE_frame_<ARM>.json for one arm from environment variables set by frame_arm.sh."""
import glob
import hashlib
import json
import os
import platform
import time

E = os.environ


def sha(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


rd, ev, code = E["RD"], E["EV"], E["CODE"]
replay = E.get("REPLAY") == "1"
preds = sorted(glob.glob(ev + "/preds/*_preds*.npz"))
prov = dict(
    arm=E["ARM"], frac=E["FRAC"], frame=E["FRAME"], replay=replay, urdu_share=float(E.get("QU", "0.25")),
    urdu_share_end=E.get("QU_END") or None, curriculum_start_step=E.get("CUR_START") or None,
    train_steps=int(E.get("STEPS", "10000")), save_every=int(E.get("SAVE", "1000")), gate_step=E.get("GATE_STEP") or None,
    train_seed=int(E.get("TRAIN_SEED", "20261016")), sampling_seeds=E.get("SEEDS", "0 1 2").split(),
    recipe_reference="Route A LUO-FT (M5 m5_ft.sh): init LUO ms6 model+EMA, batch 128 x accum 4, lr 8e-5, ckpt every 1k, val1000 lowest native LDTW selection",
    recipe_actual=dict(learning_rate=(jl(rd + "/run_config.json") or {}).get("arguments", {}).get("learning_rate"),
                       batch_size=(jl(rd + "/run_config.json") or {}).get("arguments", {}).get("batch_size"),
                       gradient_accumulate_every=(jl(rd + "/run_config.json") or {}).get("arguments", {}).get("gradient_accumulate_every"),
                       env_LR=E.get("LR"), env_BATCH=E.get("BATCH"), env_ACCUM=E.get("ACCUM")),
    init_from=dict(path=E["INIT"], sha256=sha(E["INIT"])), urdu_npz=dict(path=E["URDU"], sha256=sha(E["URDU"])),
    replay_npz=dict(path=E["REPLAY_NPZ"], sha256=sha(E["REPLAY_NPZ"])) if replay else None,
    selected_milestone=int(open(ev + "/best_milestone.txt").read()) if os.path.exists(ev + "/best_milestone.txt") else None,
    selection=jl(ev + "/selection.json"), gate=[jl(f) for f in sorted(glob.glob(ev + "/gate_*.json"))],
    init_from_json=jl(rd + "/init_from.json"), replay_config=jl(rd + "/replay_config.json"), replay_stats=jl(rd + "/replay_stats.json"),
    run_config=jl(rd + "/run_config.json"), training_complete=jl(rd + "/training_complete.json"), gate_fail=jl(rd + "/GATE_FAIL"),
    code_dir=code, code_sha256sums=open(code + "/SHA256SUMS.txt").read().splitlines(),
    preds={os.path.basename(f): sha(f) for f in preds}, sidecars=sorted(os.path.basename(f) for f in glob.glob(ev + "/preds/*_preds*.json")),
    host=platform.node(), slurm_job=E.get("SLURM_JOB_ID"), gpu=os.popen("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null").read().strip() or None,
    smoke_units=E.get("SMOKE_UNITS") or None,
    protocol="uhwr_eval 0.3.1 native + standardised via e0_score.py; Urdu sets sampled in the arm's frame and inverted before scoring (frame_sample.py); LTR/German sets identity",
    at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
json.dump(prov, open(ev + f"/PROVENANCE_frame_{E['ARM']}.json", "w"), indent=1)
print("PROVENANCE written", ev + f"/PROVENANCE_frame_{E['ARM']}.json")
