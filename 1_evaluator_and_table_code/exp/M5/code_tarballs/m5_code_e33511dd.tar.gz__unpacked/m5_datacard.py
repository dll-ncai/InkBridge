"""Write DATA_CARD.md and PROVENANCE_m5_multiscript_v1.json for data/m5_multiscript_v1 from the build outputs.
python m5_datacard.py --out DATA_DIR --code-dir CODE_DIR --code-tarball TGZ --env-dir ENV_DIR --command "..."
"""
import argparse
import glob
import hashlib
import json
import os
import time

import pandas as pd


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


LICENCE = {
    "casia_zh": "CASIA-OLHWDB 1.0/1.1 (NLPR CASIA): free for academic research under the CASIA licence agreement; no redistribution",
    "casia_en": "CASIA-OLHWDB 1.0/1.1 (NLPR CASIA): free for academic research under the CASIA licence agreement; no redistribution",
    "hpl_tamil": "hpl-tamil-iso-char (HP Labs India): research use only; no publishing/redistribution of any part (data/hpl_tamil/README.md)",
    "iam": "IAM-OnDB (FKI/IAM): non-commercial research; no redistribution (data/d2_v2/iam/DATA_CARD.md)",
    "ouhdl": "OUHD-L: terms as in data/ouhdl_ligatures_v3 data card",
}
SOURCE = {
    "casia_zh": "CASIA OLHWDB1.0+1.1 train .pot (data/casia/CASIA/train, 576 writer files), labels = single CJK char (U+4E00-9FFF)",
    "casia_en": "same .pot files, labels = single ASCII letter a-z/A-Z (D2 casia_en rule)",
    "hpl_tamil": "HPL IWFHR06 train online (data/hpl_tamil/extracted/hpl-tamil-iwfhr06-train-online-0.2/new/usr_<id>)",
    "iam": "data/d2_v2/iam (X3 component units of IAM-OnDB t1 lines): train = e0_train.npz (trainset), val = e0_val1.npz (testset_v)",
    "ouhdl": "data/ouhdl_e0_v3 (E0 arrays of OUHD-L ligature units v3.0): train = e0_train.npz, val = e0_val.npz",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--code-dir", required=True); ap.add_argument("--code-tarball", required=True)
    ap.add_argument("--env-dir", required=True); ap.add_argument("--command", required=True)
    a = ap.parse_args()
    st = json.load(open(os.path.join(a.out, "build_stats_M5.json")))
    sq = pd.read_csv(os.path.join(a.out, "seqlen_check_M5.csv"))
    ft = json.load(open(os.path.join(a.out, "ft_subsets_M5.json")))
    files = {os.path.basename(f): dict(sha256=sha(f), bytes=os.path.getsize(f)) for f in sorted(glob.glob(os.path.join(a.out, "*")))
             if os.path.isfile(f) and not os.path.basename(f).startswith(("DATA_CARD", "PROVENANCE_"))}
    prov = dict(dataset="m5_multiscript_v1", built_by="M5 (X6) prep job", status="full" if not st["params"]["limit_files"] else "smoke",
                built_utc=st["finished_utc"], started_utc=st["started_utc"], seconds=st["seconds"],
                slurm=dict(job_id=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename, partition=os.environ.get("SLURM_JOB_PARTITION")),
                command=a.command, params=st["params"], seed=st["seed"], inputs=st["inputs"], pools=st["pools"],
                errors=st["errors"], casia_val_writers=st["casia_val_writers"], tamil_val_users=st["tamil_val_users"],
                ft_subsets=ft, code=dict(dir=a.code_dir, tarball=a.code_tarball, tarball_sha256=sha(a.code_tarball),
                                        files={os.path.basename(f): sha(f) for f in sorted(glob.glob(os.path.join(a.code_dir, "*.py")))}),
                environment={os.path.basename(f): sha(f) for f in sorted(glob.glob(os.path.join(a.env_dir, "*"))) if os.path.isfile(f)},
                outputs=files, licences=LICENCE, redistributable=False)
    json.dump(prov, open(os.path.join(a.out, "PROVENANCE_m5_multiscript_v1.json"), "w"), indent=1, default=str)

    L = []
    L.append("# DATA CARD: m5_multiscript_v1 (M5 / X6 multi-script training set)\n")
    L.append(f"- **Built:** {st['finished_utc']} on CLUSTER-A ({os.uname().nodename}, Slurm {os.environ.get('SLURM_JOB_ID')}), status: {prov['status']}. Provenance: PROVENANCE_m5_multiscript_v1.json.")
    L.append("- **Purpose:** equal-size training sets per script for the M5 joint model (5 scripts) and the leave-Urdu-out model (4 scripts), "
             "plus selection val sets, an IAM test sample and fixed-seed OUHD-L fine-tune subsets.")
    L.append("- **Licences:** every source forbids redistribution; this derived set stays on CLUSTER-A and is not an artifact.")
    for s, t in LICENCE.items():
        L.append(f"  - {s}: {t}")
    L.append("\n## Unit format (identical for all scripts; E0 format)")
    L.append("Ink on the 64x64 evaluation canvas (uhwr_eval.fit_transform_unit: bbox fitted into 56x56, centred, 4 px margin, y down, x not mirrored). "
             "CASIA / Tamil raw ink first passes uhwr_eval.clean_ink (D2 UF.to_eval_space 'isolated'); IAM and OUHD-L arrays are taken unchanged "
             "from data/d2_v2/iam and data/ouhdl_e0_v3. Target `seq` = e0_adapter.encode: per-stroke 1 px arc-length resampling, step x1.05 until <= 320 points "
             "(no truncation), Nagamatsu [x, y, pen, eos]. Condition `cond_rendered` = e0_adapter.render_condition (2 px anti-aliased, 64x64, black on white uint8). "
             "`sample_ids` = writer key `<script>:<writer>` (e0_score.py rows then carry writer clusters); `writer_ids` = integer index of the sorted writer keys within the file; `scripts` = script tag.")
    L.append("\n## Sources and sampling (seed 20261016)")
    for s, t in SOURCE.items():
        L.append(f"- **{s}:** {t}")
    L.append(f"- Val writers: CASIA {len(st['casia_val_writers'])} of the train writer files ({', '.join(st['casia_val_writers'])}); "
             f"Tamil {len(st['tamil_val_users'])} users ({', '.join(st['tamil_val_users'])}); IAM / OUHD-L official val splits. "
             "Train samples come only from non-val writers; val samples only from val writers (CASIA / Tamil).")
    L.append("- Train sample: n_train units uniformly at random without replacement from the train pool; val sample: n_val units from the val pool.")
    L.append("\n| pool | pool units | pool writers | picked |\n|---|---|---|---|")
    for k, v in st["pools"].items():
        L.append(f"| {k} | {v['pool_units']} | {v.get('pool_writers', 'n/a (all)')} | {v['picked']} |")
    L.append("\n## Point counts at the E0 setting (64x64 canvas, 1 px arc-length step, L = 320): equal-sequence-length check")
    L.append("n_points_1px = points after 1 px per-stroke resampling before any step growth; n_seq = stored sequence length (<= 320); "
             "step = resampling step used (1.0 unless n_points_1px > 320). Computed on the sampled units (train = the 20k training sample of each script). "
             "OUHD-L train arrays carry no GT ink: for its units with step > 1 the 1 px count is re-estimated from the decoded sequence (n1px_exact_units says how many are exact).")
    cols = ["script", "role", "n_units", "n_writers", "n_errors", "n1px_exact_units", "n_points_1px_p50", "n_points_1px_p90", "n_points_1px_p99",
            "n_points_1px_p100", "n_points_1px_mean", "frac_n1px_gt_320", "n_seq_mean", "n_seq_p100", "step_p90", "step_p100", "frac_step_gt_1",
            "n_strokes_p50", "n_strokes_mean"]
    L.append("\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for _, r in sq.iterrows():
        L.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    L.append("\n## Files")
    for k, v in files.items():
        L.append(f"- `{k}`: {v['bytes']} B, sha256 {v['sha256']}")
    L.append("\n## Fine-tune subsets (OUHD-L E0 train, nested)")
    L.append(f"Rule: {ft['rule']}. Source {ft['source']} (sha256 {ft['source_sha256']}).")
    for k, v in ft["subsets"].items():
        L.append(f"- {k}: {v['n_units']} units, {v['n_writers']} writers, {v['n_lines']} lines -> {v['file']}")
    L.append("\n## Eval sets used by M5 (not rebuilt here)")
    L.append("OUHD-L v3 test (data/ouhdl_e0_v3/e0_test.npz, 4,406 units) and gold (e0_gold.npz, 354 units); X4a 500-unit sets of CASIA zh/en "
             "(OLHWDB1.1 test writers) and HPL Tamil (IWFHR06 test) in exp/X4a/data; m5_test_iam500.npz (500 IAM testset_f units with a known writer, seed 2026).")
    open(os.path.join(a.out, "DATA_CARD.md"), "w").write("\n".join(L) + "\n")
    print("wrote DATA_CARD.md and PROVENANCE_m5_multiscript_v1.json")


if __name__ == "__main__":
    main()
