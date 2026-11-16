# DATA_CARD: X4a zero-shot input arrays (x4a_casia_zh.npz, x4a_casia_en.npz, x4a_hpl_tamil.npz)

- Built: 19 Oct 2026 by build_x4a.py (e0q_code bundle, $U/exp/e0q/code), CLUSTER-A Slurm 24480348 (agent job 55d39509).
- Source: exp/D2/data/d2_pilot_v1 (d2_pilot_v1.npz + d2_pilot_v1_index.csv; 500 units per set, selected with seed 2026 for D2). Sets: casia_zh, casia_en (CASIA OLHWDB1.1 test), hpl_tamil (HPL IWFHR06 test). All isolated units.
- Licence: CASIA and HPL licences apply; arrays stay on CLUSTER-A and are not artifacts.
- Units / writers: casia_zh 500 / 60; casia_en 500 / 60; hpl_tamil 500 / 52 (x4a_build_stats.json).
- Ground truth: UF.to_eval_space (clean_ink + fit_transform_unit: 56 px box, 4 px margin, 64x64), i.e. the D2 evaluation space. Stored as gt_points / gt_stroke_id / gt_offsets.
- Conditions: cond_rendered = E0 training renderer (uhwr_eval.render_image 64x64, 2 px, anti-aliased, black on white uint8); cond_d2 = D2 renderer (unit_features.render_input: 256x256, 8 px, anti-aliased) down-sampled to 64x64 with cv2.INTER_AREA.
- seq / seq_offsets / step: e0_adapter.encode target sequences (needed by OuhdlE0Dataset; not used in sampling). Units whose 1 px resampling exceeded 320 points got a larger step (casia_zh 243, casia_en 0, hpl_tamil 4 units).
- sample_ids = writer_id string (so e0_score.py summary CIs cluster on writers); writer_ids = integer index of the sorted writer ids within the set; set_ids, labels, directions copied from the index.
- sha256: x4a_casia_zh.npz 93f4efcb99941a77fd942e19e9001316d14654ac77e3dc5b622a6da2cdbc0f72; x4a_casia_en.npz eb8c6a7a54fac56448b5232e2ef56726a9046198ba439129241bf3d9d9af7d6a; x4a_hpl_tamil.npz d868ba858f07de2e5d179f5a635a4007fbfc25b1fa4bdbb1149ccf073dce6938.
- Provenance: ../PROVENANCE_X4a.json.


## Licence

LICENSE: not redistributable (CASIA-OLHWDB licence agreement; HP Labs India Tamil licence: no redistribution). CLUSTER-A only.

(Licence line added 2026-10-22 by the uhwr licence pass; register: shared/licences/licence_register_20261022.csv; dataset: data.)
