# Loader for the reduced per-unit score files (numpy + pandas).
import glob, os, numpy as np, pandas as pd
def load(folder, set_name=None):
    """Return one row per (scored file, unit): set, tag, seed, unit_id, writer_id, error and six metrics (native protocol)."""
    F = pd.read_csv(os.path.join(folder, 'FILES_INDEX.csv')).set_index('file_idx')
    out = []
    for p in sorted(glob.glob(os.path.join(folder, 'rows__*.npz'))):
        s = os.path.basename(p)[:-4].split('__rows__')[-1]
        if set_name and s != set_name: continue
        z = np.load(p); U = pd.read_csv(os.path.join(folder, 'UNITS_%s.tsv' % s), sep='\t', dtype=str).set_index('unit_idx')
        d = pd.DataFrame(z['metrics'], columns=[str(m) for m in z['metric_names']])
        d.insert(0, 'error', z['error']); d.insert(0, 'unit_idx', z['unit_idx']); d.insert(0, 'file_idx', z['file_idx'])
        d['unit_id'] = U.unit_id.reindex(d.unit_idx.astype(str)).to_numpy(); d['writer_id'] = U.writer_id.reindex(d.unit_idx.astype(str)).to_numpy()
        d = d.join(F[['set', 'tag', 'seed', 'dir']], on='file_idx'); out.append(d)
    return pd.concat(out, ignore_index=True)
# Per-unit table used by the table scripts: mean over the sampling seeds of one checkpoint tag:
#   t = load(folder, 'test_ouhdl'); u = t[t.tag == TAG].groupby('unit_id')[METRICS].mean()
# Pooled over training seeds: mean of the per-unit tables of the three training-seed tags.
