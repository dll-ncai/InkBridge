# SPEC_router336 — script router with the Urdu class trained on the 336-unit 1% subset

Written 2026-11-15T13:29:15Z, before any router336 training run and before any router336 checkpoint is scored on a test manifest.
Issue: L07 (ED-06, NO-03) of the v10 review. Track owner: router agent, RV11. Run dir: `experiments/RV11/router336` (Mac).

## 1. Reported router (records read, nothing re-run yet)
- Checkpoint `experiments/D3_ROUTE/router/script_clf.pt`, sha256 22bb8802f2da46fc37972eec430a1439ad830905712f22f6d923c38829db45aa, built 2026-10-24 on the Mac CPU, env bp2-torch.
- Code `script_clf.py` sha256 58710f5729d71d6775078351fa6c5993b2a98aeb413ff333edc01ee8ed2b962f (tar 7126a969...).
- Model: ScriptCNN, 4x[conv3x3-BN-ReLU-maxpool] widths 16/32/64/96, dropout 0.3, fc128, fc5; 276,485 parameters; input 64x64 `cond_rendered` (uint8, ink inverted to 1).
- Train data: `m5_joint_train.npz` (CLUSTER-A sha256 56649a25...; cond extract sha256 ebfd07f5...), 100,000 units = 20,000 each of ouhdl, casia_zh, casia_en, hpl_tamil, iam. Labels: script name per unit.
- Recipe (train_log.json args): AdamW lr 2e-3, weight decay 1e-4, OneCycleLR over epochs x steps, batch 256, 8 epochs, patience 8, random roll shift <=3 px per batch, seed 0, 12 threads. Checkpoint = epoch with best 5-class accuracy on m5_val_* (5 x 250 units); best epoch 6, val acc5 0.9832.
- Paper routing rule (v10 tex, notation table): a unit goes to the fine-tuned (Urdu) model iff P(target script) >= 0.5; all other units go to the backbone.
- Reported test read (accuracy_per_set.csv): acc_script ouhdl 0.99637, casia_zh 0.980, casia_en 0.960, hpl_tamil 0.994, iam500 0.964; German sets have no script label.

## 2. What changes
Only the Urdu training rows. The 20,000 ouhdl rows of m5_joint_train are dropped and replaced by the 336 units of the 1% subset (p01). The 80,000 non-Urdu rows, the model, optimiser, schedule, augmentation, epochs, val sets and checkpoint rule are unchanged. No threshold tuning and no Mahalanobis gate: the decision is P(ouhdl) >= 0.5 as in the paper.

Consequences of the change that are not choices: arm A has 80,336 training rows, so 314 steps per epoch (reported: 391). Threads are set to 4 per run (reported: 12) so that three runs fit on the machine; thread count is recorded in each train log.

## 3. The 336 units
- Definition: `experiments/M5/data_build/ft_subsets_M5.json` (sha256 71f85df0...): sorted first 336 indices of numpy default_rng([20261016, 9]).permutation(33615) over OUHD-L E0 train (`ouhdl_e0_v3/e0_train.npz`, sha256 2c4ddc48...; the Mac copy `experiments/e0/data/e0_train.npz` has the same sha256). 183 writers.
- IDs: `p01_unit_ids.txt` in the run dir; sha256 of the newline-joined IDs 54e4124ca09ffdde7d2e44f397fc1fee6a30647bc19349bad1a1ed8f73b1e65c; sha256 of the 336 images (uint8 bytes) 2ddb21e9b93cea1dcc299770ab1684d8c8271814cc8c53653690e22aeeade066. Both equal the values computed on CLUSTER-A today from `data/m5_multiscript_v1/m5_ft_ouhdl_p01.npz` (file sha256 0e71475d...), the file the 1% fine-tunes use.
- Overlap with m5_joint_train unit IDs: 0 of 336 (checked today on the pulled extract).

## 4. Arms (class imbalance 336 vs 20,000)
- Arm A, primary, `natural`: reported recipe unchanged on 80,000 non-Urdu + 336 Urdu rows; each epoch is one permutation of the 80,336 rows. No re-weighting, no oversampling.
- Arm B, the single pre-specified balanced arm, `balanced`: each epoch = all 80,000 non-Urdu rows + 20,000 draws with replacement from the 336 (100,000 rows, 391 steps), i.e. the Line R day-1 recipe (`rr_train.py`).
- No other arm. The arm quoted in the paper is A; B is reported next to it as a labelled sensitivity. No arm is chosen on test results.

## 5. Seeds
0, 1, 2 for each arm (6 trainings). torch.manual_seed and np.random.seed as in script_clf.py. Seed 0 is the reported router's seed.

## 6. Checkpoints
Primary: best val acc5 on m5_val_* (5 x 250), as in the reported recipe. Disclosure: this selection set holds 250 labelled Urdu val units (m5_val_ouhdl; shares 18 of 31 writers with p01 per the Line R day-1 record). The last-epoch (epoch 7) checkpoint is saved too and scored as a secondary read that uses no Urdu val labels.

## 7. Test sets and metrics
Seven test sets (image extracts pulled 2026-10-24, sha256 in pull_SHA256SUMS.txt): test_ouhdl 4,406; test_casia_zh 500; test_casia_en 500; test_hpl_tamil 500; test_iam500 500; german_unipen_de_test 827; german_onhw_all 5,925.
Per set, per arm, per seed, and for the reported router:
- n_routed_urdu = number of units with P(ouhdl) >= 0.5.
- route_acc = share of units sent to the correct side of the binary route (Urdu set: routed; the six other sets: not routed), with a Wilson 95% interval.
- acc_script5 = 5-class argmax accuracy (five labelled sets; not defined for the German sets).
- n_route_changed = units whose binary route differs from the reported router (paired on unit_id), split into gained and lost; exact McNemar p.
- identical = 1 if n_route_changed = 0.
Pooled rows (per arm and set): mean, min and max over the three seeds of route_acc and n_routed_urdu, and the number of seeds with identical routing.
Pipeline check, run first: the reported checkpoint is re-scored with the same evaluation code and must reproduce the P(ouhdl) >= 0.5 counts derived from the stored manifests (`manifest_*.csv`) on all seven sets; if it does not, stop.
Supplementary row, not pooled: the Line R day-1 checkpoint (CLUSTER-A, 2026-10-29, sha256 c3e0a29c..., arm-B recipe, seed 0, torch 2.7.1), scored with the same code.

## 8. What is already known (disclosed)
- The reported router's test numbers (section 1).
- Line R day 1 (29 Oct): an arm-B-recipe seed-0 router was trained on CLUSTER-A CPU. Its recorded reads are val reads (router_day1_table.csv): Urdu val1000 983/1000 routed (reported router 994/1000), LTR val250x4 0/1000, UNIPEN-de-train 90/3,927 (reported 170/3,927), best val acc5 0.980. That table holds no test-set rows.
- Arm A has not been run before according to the records read.

## 9. How the result will be read (fixed now)
1. Per set: if routing is identical to the reported router in all three seeds of arm A, the composed result on that set is unchanged by the router swap, and this is stated. Otherwise the number of units whose route changes is stated per seed.
2. Urdu test: the three-seed mean route_acc of arm A is compared with the reported router's value. A difference of at most 0.010 absolute is described as "within one point"; a larger drop is described as "lower by more than one point". 0.010 is a descriptive convention set here, not a certified margin.
3. Non-Urdu sets: for each of the six sets, the three-seed mean n_routed_urdu of arm A is stated next to the reported router's count; "no increase" is used only if the mean is not above the reported count.
4. Seed spread (min, max) is always reported with the mean. Arm B and the last-epoch read are reported in the same table and labelled; they do not replace arm A.
5. Composed metrics (Table 8, retention) are not recomputed unless per-unit predictions of both the fine-tuned model and the backbone are already on hand for the units whose route changes; without them only the count of changed-route units is reported.
6. If a training run fails or diverges (val acc5 < 0.5 at the best epoch), the row is kept and marked; no seed is replaced.

## 10. Where it runs and what is written
Local Mac CPU, conda env bp2-torch (the reported router's env). No CLUSTER-A/CLUSTER-B GPU. Inputs were copied read-only from CLUSTER-A today (new dir `tmp/rv11_router336_pull_20261115`, tar sha256 416283b3...). Outputs: `runs/<arm>_s<seed>/` (checkpoints, train log), `eval/` (per-unit manifests), `router336_results.csv`, `router336_checkpoints_sha256.csv`, `PROVENANCE_router336.json`.
