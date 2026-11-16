# uhwr_baselines (OUHD-L adapter + baseline runners)

* `ouhdl.py`   - OUHD-L v1.0 loader (Zenodo split manifests), raw CSV -> shared stroke format, line units and
                 connected-component (ligature) units with diacritic assignment.  sample_id = <series>_<board>_<line>.
* `inputs.py`  - input conditions: `rendered` (uniform-width rendering of GT strokes) and `dataset` (OUHD-L img/ PNG),
                 each with the affine map back to the evaluation canvas.
* `geometric.py` - skeleton/graph baseline with right-to-left prior (dots last / dots after nearest body).
* `momna.py`   - wrapper around dll-<site>/Synthetic-Stroke-Trajectory Stages 1-3 (PathBExtractor, unchanged model code).
* `inksight_runner.py` - InkSight Small-p zero-shot inference (google-research/inksight notebook procedure).
* `run_ouhdl.py` - runs geometric + Momna on all test lines and scores with uhwr_eval (multiprocessing).
* `score_inksight.py`, `aggregate.py` - InkSight scoring and aggregation (mean, SD, cluster bootstrap CI).
Data download: `fetch_parts.sh`/`fetch_one.sh` (parallel ranged download; sha256 of the core tar checked against
Zenodo SHA256SUMS: d2cdf97bd9ba4ed645ed5bcdd85be0412036113fe7d791bd7518dec5465278ae).
Requires uhwr_eval 0.1.0 and external repos cloned under ext/ (momna, inksight).
