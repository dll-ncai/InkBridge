# SPEC_reviewer_archive_v11 (written 2026-11-15, before any collection run)
Purpose: reviewer archive for InkBridge v11, issue L09 (ED-05, RE-03, RE-04).
Scope: copies of existing records only. No training, sampling or scoring is run. No number in the paper is recomputed.
Sources: cluster project root ($U) and the local project folder (read only).
Steps:
1. Stage 1 (cluster login node, python3 stdlib + numpy if present): copy listed code, configs, plans, table CSVs and data records
   (per-file cap 3 MB unless allow-listed) into a NEW directory $U/tmp/rv11_archive/stage1; record source path, sha256, size, mtime
   per file in SOURCES.tsv and every file over the cap in SKIPPED.tsv; dump 1-D ID arrays of the subset npz files to CSV (no ink arrays);
   list per-unit rows directories. Output: one tar.gz, fetched once.
2. Stage 2: per-unit score rows behind Tables 5, 6, 8, 10, 11, 12 first, then others while the archive stays under ~100 MB.
   Rows are column subsets or byte copies of the scored rows files; no averaging, no rounding. Source sha256 recorded per file.
3. Local assembly: add local OUHD-L division v3.0 records, router manifests, evaluator release; scrub names, usernames, e-mail
   addresses, host names, absolute host paths and agent/model identifiers; verify by grep; write README, MANIFEST.sha256, PROVENANCE.
Sharing limits: ground-truth ink for OUHD-L only; CASIA, IAM, UNIPEN, HPL Tamil, OnHW: IDs only, no ink, no rendered images. No model weights.
Nothing on the cluster or in the local project folder is deleted or overwritten. Nothing is uploaded, pushed or published.
