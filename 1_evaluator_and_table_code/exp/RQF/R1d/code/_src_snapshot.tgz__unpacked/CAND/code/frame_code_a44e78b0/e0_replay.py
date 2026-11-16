"""Replay loader + in-job milestone gate for the E0/M5 fine-tune path (C1 / C3 of uhwr_method_candidates_2026-10-27).

Two-source batches.  Every DataLoader batch of size B holds k_u = round(q_u * B) units from the low-resource
("Urdu") subset npz and B - k_u units from the replay shards of a multi-script npz (m5_luo_train.npz: 4 LTR
scripts, key `scripts`), where the replay script of each unit is drawn with probability
    p_s = n_s^(1/5) / sum_s' n_s'^(1/5)        (temperature-capped mix; equal shards -> uniform)
and the unit within a script comes from a per-script shuffled-epoch stream (without replacement inside an epoch,
reshuffled every epoch, all streams seeded from one integer).  The Urdu share can follow a curriculum:
q_u(step) = q_u for step < curriculum_start, then linear from q_u to q_u_end up to train_steps.  "step" is the
optimiser step = (batches consumed) // gradient_accumulate_every, which is exact for one process; with
accelerate split_batches and P processes the loader batch is shared across the P processes, so the sampler
yields P*B indices per optimiser micro-step and counts accordingly (num_processes is passed in).

Canonical frame.  The frame (e0_frames.py) is applied by the datasets themselves (OuhdlE0Dataset(frame=...,
frame_scripts=...)); the sampler only mixes indices.  In the fine-tune path the frame is applied to the Urdu
subset only (script tag 'ouhdl'); the LTR replay units stay in the identity frame.

Gate.  gate_eval(...) samples (DDIM, EMA weights) OUHD-L gold (354 units, frame applied to the condition and
inverted on the output) and 200 UNIPEN-de test units (identity frame; e0_sample.py --subset 200 --subset-seed 0
rule: np.sort(default_rng(0).choice(n, 200, replace=False))), scores both with uhwr_eval 0.3.1 native protocol
against the native GT ink (gt_points) exactly like e0_score.py, and writes gate_<k>k.json (step = k*1000, e.g. gate_4k.json;
otherwise gate_step<step>.json) with
{gold_aiou, gold_tau, unipen_direction, pass, thresholds, ...}.  Means are nan-means over units (pandas
DataFrame.mean semantics of e0_score.py summary).  Preds are also written in the Route A layout
(<set>__<label>__seed<s>_preds.npz + .json sidecar) so the gate read can be re-scored later.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Sampler

import e0_adapter as A
import e0_frames as F
from e0_dataset import OuhdlE0Dataset

GATE_THRESHOLDS = {"gold_aiou_min": 0.750, "gold_tau_min": 0.80, "unipen_direction_min": 0.90}
URDU_TAG = "ouhdl"


def sha256_file(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def script_probs(counts: dict, power: float = 0.2) -> dict:
    """p_s proportional to n_s ** power (power = 1/5)."""
    names = sorted(counts)
    w = np.array([counts[s] ** power for s in names], np.float64)
    w /= w.sum()
    return dict(zip(names, w.tolist()))


class _EpochStream:
    """Infinite stream over a fixed index array: shuffled epochs without replacement."""

    def __init__(self, indices, rng):
        self.idx = np.asarray(indices, np.int64)
        if len(self.idx) == 0:
            raise ValueError("empty source")
        self.rng = rng
        self.perm = self.rng.permutation(self.idx)
        self.pos = 0
        self.epochs = 0

    def take(self, k):
        out = []
        while k > 0:
            if self.pos >= len(self.perm):
                self.perm = self.rng.permutation(self.idx)
                self.pos = 0
                self.epochs += 1
            m = min(k, len(self.perm) - self.pos)
            out.append(self.perm[self.pos:self.pos + m])
            self.pos += m
            k -= m
        return np.concatenate(out)


class ReplayBatchSampler(Sampler):
    """Yields index lists into ConcatDataset([urdu_ds, replay_ds]) (replay indices offset by len(urdu_ds))."""

    def __init__(self, n_urdu, urdu_train_indices, replay_indices_by_script, batch_size, q_u, q_u_end=None,
                 curriculum_start_step=None, train_steps=10_000, gradient_accumulate_every=4, seed=0, start_step=0,
                 num_processes=1):
        self.n_urdu = int(n_urdu)
        self.B = int(batch_size) * int(num_processes)
        self.q_u, self.q_u_end = float(q_u), (None if q_u_end is None else float(q_u_end))
        self.cur_start = curriculum_start_step
        self.train_steps = int(train_steps)
        self.ga = int(gradient_accumulate_every)
        self.seed, self.start_step = int(seed), int(start_step)
        self.rng = np.random.default_rng([self.seed, 0x5E9])
        self.urdu = _EpochStream(urdu_train_indices, np.random.default_rng([self.seed, 1]))
        self.scripts = sorted(replay_indices_by_script)
        self.streams = {s: _EpochStream(np.asarray(replay_indices_by_script[s]) + self.n_urdu,
                                        np.random.default_rng([self.seed, 2 + i])) for i, s in enumerate(self.scripts)}
        self.counts = {s: int(len(replay_indices_by_script[s])) for s in self.scripts}
        self.p = script_probs(self.counts)
        self.p_vec = np.array([self.p[s] for s in self.scripts])
        self.batches_yielded = 0
        self.carry = 0.0
        self.log = {"k_u_hist": {}, "script_draws": {s: 0 for s in self.scripts}, "urdu_draws": 0}

    def urdu_share(self, step):
        if self.q_u_end is None or self.cur_start is None or step < self.cur_start:
            return self.q_u
        span = max(self.train_steps - self.cur_start, 1)
        t = min(max((step - self.cur_start) / span, 0.0), 1.0)
        return self.q_u + t * (self.q_u_end - self.q_u)

    def current_step(self):
        return self.start_step + self.batches_yielded // self.ga

    def __len__(self):  # a nominal epoch; the trainer cycles the loader anyway
        return max(1, (self.train_steps - self.start_step) * self.ga)

    def __iter__(self):
        while True:
            step = self.current_step()
            q = self.urdu_share(step)
            # error-diffusion rounding: exact k_u when q*B is an integer (32 of 128 at q=0.25), long-run share q otherwise
            self.carry += q * self.B
            k_u = int(np.floor(self.carry + 1e-9))
            k_u = min(max(k_u, 0), self.B)
            self.carry -= k_u
            idx_u = self.urdu.take(k_u) if k_u else np.zeros(0, np.int64)
            n_r = self.B - k_u
            if n_r:
                draws = self.rng.choice(len(self.scripts), size=n_r, p=self.p_vec)
                parts = []
                for i, s in enumerate(self.scripts):
                    m = int((draws == i).sum())
                    if m:
                        parts.append(self.streams[s].take(m))
                        self.log["script_draws"][s] += m
                idx_r = np.concatenate(parts)
            else:
                idx_r = np.zeros(0, np.int64)
            batch = np.concatenate([idx_u, idx_r])
            self.rng.shuffle(batch)
            self.log["urdu_draws"] += k_u
            self.log["k_u_hist"][k_u] = self.log["k_u_hist"].get(k_u, 0) + 1
            self.batches_yielded += 1
            yield batch.tolist()

    def describe(self):
        return dict(batch_size_total=self.B, q_u=self.q_u, q_u_end=self.q_u_end, curriculum_start_step=self.cur_start,
                    train_steps=self.train_steps, gradient_accumulate_every=self.ga, seed=self.seed, start_step=self.start_step,
                    replay_counts=self.counts, replay_probs=self.p, replay_prob_rule="p_s ∝ n_s^(1/5)",
                    n_urdu_train=int(len(self.urdu.idx)), stream_rule="per-source shuffled epochs without replacement")


def build_replay_sources(urdu_npz, replay_npz, condition, frame, frame_scripts=(URDU_TAG,), replay_scripts=None,
                         urdu_train_indices=None):
    """Datasets for the two sources.  Frame is applied to the Urdu npz (tagged URDU_TAG when it has no `scripts`
    key) and, via frame_scripts, NOT to the replay shards unless their tag is listed."""
    urdu_ds = OuhdlE0Dataset(str(urdu_npz), condition, frame=frame, frame_scripts=set(frame_scripts), script_tag=URDU_TAG)
    replay_ds = OuhdlE0Dataset(str(replay_npz), condition, frame=frame, frame_scripts=set(frame_scripts))
    tags = replay_ds.scripts
    use = sorted(set(tags)) if replay_scripts is None else list(replay_scripts)
    by_script = {s: np.flatnonzero(tags == s) for s in use}
    for s, ix in by_script.items():
        if len(ix) == 0:
            raise ValueError(f"replay npz has no units with script tag {s!r}; tags present: {sorted(set(tags))}")
    if urdu_train_indices is None:
        urdu_train_indices = np.arange(len(urdu_ds))
    return urdu_ds, replay_ds, by_script, np.asarray(urdu_train_indices)


def make_replay_loader(urdu_ds, replay_ds, by_script, urdu_train_indices, *, batch_size, seq_length, q_u, q_u_end,
                       curriculum_start_step, train_steps, gradient_accumulate_every, seed, start_step, num_workers,
                       pin_memory, num_processes=1):
    from functools import partial
    from utils.casia_grid import collate_fn
    ds = ConcatDataset([urdu_ds, replay_ds])
    sampler = ReplayBatchSampler(len(urdu_ds), urdu_train_indices, by_script, batch_size, q_u, q_u_end, curriculum_start_step,
                                 train_steps, gradient_accumulate_every, seed, start_step, num_processes)
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=partial(collate_fn, sequence_length=seq_length),
                    num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)
    return dl, sampler


# ----------------------------------------------------------------------------------------------------------------
# gate
# ----------------------------------------------------------------------------------------------------------------
def subset_indices(n, k, seed):
    """e0_sample.py --subset k --subset-seed seed rule."""
    return np.sort(np.random.default_rng(seed).choice(n, min(k, n), replace=False))


@torch.no_grad()
def sample_dataset(model, ds, device, batch_size, seed, frame="identity"):
    """DDIM-sample every unit of ds in index order (e0_sample.py loop), then apply the inverse frame to the
    (N, L, 4) output.  Seeds torch and numpy once (routeA_sample.py protocol).  Returns float32 preds."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    out = []
    for b in range(0, len(ds), batch_size):
        items = [ds[i] for i in range(b, min(b + batch_size, len(ds)))]
        imgs = torch.stack([it["img"] for it in items]).to(device)
        (x, _), _, _ = model.sample(condition=imgs, ground_truth=None, batch_size=len(items), is_testing=True)
        out.append(x.transpose(1, 2).float().cpu().numpy())
    preds = np.concatenate(out).astype(np.float32)
    return F.inverse_frame_preds(preds, frame)


def score_preds(npz_path, preds, indices, procs=1):
    """uhwr_eval native-protocol rows for preds[k] vs native GT ink of unit indices[k] (e0_score.py _work)."""
    from uhwr_eval import score
    z = np.load(npz_path, allow_pickle=False)
    gp, gs, go = z["gt_points"], z["gt_stroke_id"], z["gt_offsets"]
    ids, sids = z["unit_ids"].astype(str), z["sample_ids"].astype(str)
    rows = []
    for k, i in enumerate(indices):
        gt = A.unpack_ink(gp, gs, go, int(i))
        try:
            pr = A.decode(preds[k])
            r = score(gt, pr, A.CANVAS, protocol="native", with_lpips=False)
            err = ""
        except Exception as e:  # keep the row; the summary counts errors
            r, err = {}, f"{type(e).__name__}: {e}"
        rows.append({"unit_id": ids[i], "sample_id": sids[i], **r, "error": err})
    return rows


def _nanmean(rows, key):
    v = np.array([r.get(key, np.nan) for r in rows], np.float64)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan"), int(np.isfinite(v).sum())


def gate_eval(model, device, *, gold_npz, unipen_npz, frame, out_dir, label, step, seed=0, batch_size=256,
              sampling_timesteps=50, n_unipen=200, unipen_subset_seed=0, thresholds=GATE_THRESHOLDS,
              max_units=None, checkpoint=None, extra_meta=None):
    """Run the K3 milestone gate.  max_units caps BOTH sets (CPU smoke only; recorded in the json)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    # gold: all 354 units, frame applied to the condition, inverted on the output
    n_gold = len(np.load(gold_npz, allow_pickle=False)["unit_ids"])
    gold_idx = np.arange(n_gold if max_units is None else min(max_units, n_gold))
    gold_ds = OuhdlE0Dataset(str(gold_npz), "rendered", indices=gold_idx, frame=frame, frame_scripts={URDU_TAG}, script_tag=URDU_TAG)
    gold_preds = sample_dataset(model, gold_ds, device, batch_size, seed, frame)
    gold_rows = score_preds(gold_npz, gold_preds, gold_idx)
    # UNIPEN-de: 200 units by the e0_sample subset rule, identity frame
    n_de = len(np.load(unipen_npz, allow_pickle=False)["unit_ids"])
    de_idx = subset_indices(n_de, n_unipen, unipen_subset_seed)
    if max_units is not None:
        de_idx = de_idx[:max_units]
    de_ds = OuhdlE0Dataset(str(unipen_npz), "rendered", indices=de_idx, frame="identity")
    de_preds = sample_dataset(model, de_ds, device, batch_size, seed, "identity")
    de_rows = score_preds(unipen_npz, de_preds, de_idx)
    gold_aiou, n1 = _nanmean(gold_rows, "aiou")
    gold_tau, n2 = _nanmean(gold_rows, "kendall_tau")
    de_dir, n3 = _nanmean(de_rows, "direction_acc")
    checks = {"gold_aiou": gold_aiou >= thresholds["gold_aiou_min"], "gold_tau": gold_tau >= thresholds["gold_tau_min"],
              "unipen_direction": de_dir >= thresholds["unipen_direction_min"]}
    passed = all(checks.values()) and all(np.isfinite([gold_aiou, gold_tau, de_dir]))
    # Route A layout preds + sidecars
    files = {}
    for set_name, preds, ds, idx, fr, npz in (("gold_ouhdl", gold_preds, gold_ds, gold_idx, frame, gold_npz),
                                               ("german_unipen_de_test", de_preds, de_ds, de_idx, "identity", unipen_npz)):
        name = f"{set_name}__{label}__seed{seed}_preds" + ("" if max_units is None else f"_smoke{max_units}")
        fn = out_dir / (name + ".npz")
        meta = dict(set=set_name, label=label, step=step, frame=fr, frame_desc=F.describe(fr), inverse_applied=fr != "identity",
                    npz=str(npz), npz_sha256=sha256_file(npz), n_units=int(len(idx)), indices_rule=("all" if set_name == "gold_ouhdl" else f"e0_sample --subset {n_unipen} --subset-seed {unipen_subset_seed}"),
                    sampler="DDIM", sampling_timesteps=sampling_timesteps, ddim_eta=0.0, seed=seed, batch_size=batch_size, weights="EMA",
                    device=gpu, torch=torch.__version__, host=platform.node(), checkpoint=None if checkpoint is None else str(checkpoint),
                    checkpoint_sha256=None if checkpoint is None else sha256_file(checkpoint), max_units=max_units,
                    utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **(extra_meta or {}))
        np.savez_compressed(fn, preds=preds, unit_ids=ds.unit_ids[ds.idx], specialist=np.array([label] * len(idx)), meta=json.dumps(meta))
        meta["preds_sha256"] = sha256_file(fn)
        json.dump(meta, open(str(fn)[:-4] + ".json", "w"), indent=1)
        files[set_name] = fn.name
    import csv
    for set_name, rows in (("gold_ouhdl", gold_rows), ("german_unipen_de_test", de_rows)):
        keys = sorted({k for r in rows for k in r})
        with open(out_dir / f"{set_name}__{label}__seed{seed}_rows.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    res = dict(step=step, label=label, frame=frame, gold_aiou=gold_aiou, gold_tau=gold_tau, unipen_direction=de_dir,
               n_gold=int(len(gold_idx)), n_gold_aiou_defined=n1, n_gold_tau_defined=n2, n_unipen=int(len(de_idx)), n_unipen_direction_defined=n3,
               thresholds=dict(thresholds), checks=checks, **{"pass": bool(passed)}, protocol="uhwr_eval 0.3.1 native, nan-mean over units",
               sampler=f"DDIM-{sampling_timesteps} eta 0 seed {seed}, EMA weights", device=gpu, seconds=round(time.time() - t0, 1),
               files=files, max_units=max_units, smoke=max_units is not None)
    json.dump(res, open(out_dir / f"gate_{step // 1000}k.json" if step % 1000 == 0 else out_dir / f"gate_step{step}.json", "w"), indent=1)
    return res
