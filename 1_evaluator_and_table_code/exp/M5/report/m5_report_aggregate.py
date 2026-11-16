"""M5 report aggregation (22 Oct 2026). Inputs: exp/M5 result bundle (m5_results_bundle.tar.gz, extracted at --bundle) and the
E0 ms8 / X4a per-unit rows (--cmp dir). Outputs (--out): results_M5.csv (merged CLUSTER-A summaries), paired_M5_vs_E0.csv,
stepsplit_M5.csv, selection_M5.csv, keytable_native_M5.csv. CIs: copy of uhwr_baselines.aggregate.boot_ci (writer clusters, B=2000, seed 0)."""
import argparse, glob, json, os, re
import numpy as np, pandas as pd
METRICS = ["aiou","ldtw","dtw","strict_order_acc","kendall_tau","direction_acc","penup_f1","stroke_count_abs_err","stroke_match_recall","loop_orient_acc","chamfer","hausdorff","frechet","rmse"]
ORDER = {"strict_order_acc","kendall_tau","direction_acc","stroke_count_abs_err","stroke_match_recall","loop_orient_acc"}
ARM = {'m5_joint_ms9':'joint (5 scripts) ms9 = step 90k','m5_luo_ms6':'leave-Urdu-out ms6 = step 60k','m5_ft_p01_ms4':'FT 1% (336 units) ms4 = FT step 4k',
 'm5_ft_p10_ms10':'FT 10% (3,362) ms10 = FT step 10k','m5_ft_p30_ms10':'FT 30% (10,085) ms10 = FT step 10k','m5_ft_p100_ms9':'FT 100% (33,615) ms9 = FT step 9k',
 'e0_ms8':'E0 Urdu-only ms8 = step 80k (final eval rows)','e0_ms8_x4a':'E0 Urdu-only ms8, X4a zero-shot rows'}
def boot_ci(values, clusters, B=2000, seed=0):
    v = pd.Series(values).astype(float); ok = v.notna().to_numpy(); v, c = v[ok].to_numpy(), np.asarray(clusters)[ok]
    if len(v) == 0: return np.nan, np.nan
    uc, inv = np.unique(c, return_inverse=True); sums = np.bincount(inv, weights=v, minlength=len(uc)); cnts = np.bincount(inv, minlength=len(uc)).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(uc), size=(B, len(uc))); means = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1e-12)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
def meta(f):
    m = re.match(r'(.+?)__(.+?)__ms(\d+)_rows\.csv', os.path.basename(f)); return m.group(1), m.group(2), pd.read_csv(f, nrows=1)['method'][0]
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--bundle', required=True); ap.add_argument('--cmp', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
    X, H, O = a.bundle, a.cmp, a.out; os.makedirs(O, exist_ok=True)
    S = [pd.read_csv(f'{X}/runs/joint/eval/summary_M5_joint.csv'), pd.read_csv(f'{X}/runs/luo/eval/summary_M5_luo.csv'), pd.read_csv(f'{X}/runs/ft/summary_M5_ft.csv')]
    al = pd.concat(S).drop_duplicates(subset=['label','eval_set','input','protocol','metric']); al = al[al.label != 'm5_luo_ms6_zeroshot'].copy()
    al['model'] = al.label.map(ARM)
    al = al[['label','model','eval_set','input','protocol','metric','mean','ci95_lo','ci95_hi','ci','n_valid','n_units','n_writers','n_errors','n_empty_pred','rows_file']].reset_index(drop=True)
    al.to_csv(f'{O}/results_M5.csv', index=False)
    rows = {f: pd.read_csv(f) for f in sorted(glob.glob(f'{X}/runs/*/eval/*__*__ms*_rows.csv') + glob.glob(f'{X}/runs/ft/*/eval/*__*__ms*_rows.csv')) if not os.path.basename(f).startswith('val__')}
    E0R = {('ouhdl_test','rendered'):'test_rendered_ms8_ddim50_rows.csv',('ouhdl_test','dataset'):'test_dataset_ms8_ddim50_rows.csv',('ouhdl_gold','rendered'):'gold_rendered_ms8_ddim50_rows.csv',
           ('ouhdl_gold','dataset'):'gold_dataset_ms8_ddim50_rows.csv',('casia_zh500','rendered'):'casia_zh_rendered_ms8_ddim50_rows.csv',('casia_en500','rendered'):'casia_en_rendered_ms8_ddim50_rows.csv',
           ('hpl_tamil500','rendered'):'hpl_tamil_rendered_ms8_ddim50_rows.csv'}
    E0D = {k: pd.read_csv(f'{H}/{v}') for k, v in E0R.items()}
    pair = []
    for f, d in rows.items():
        es, inp, meth = meta(f)
        if (es, inp) not in E0D: continue
        e = E0D[(es, inp)]
        for prot in ('native', 'standardised'):
            A = d[d.protocol == prot].set_index('unit_id'); Bq = e[e.protocol == prot].set_index('unit_id'); cm = A.index
            assert len(cm.intersection(Bq.index)) == len(cm) == len(Bq)
            for m in METRICS:
                if prot == 'standardised' and m in ORDER: continue
                x = A.loc[cm, m].astype(float); y = Bq.loc[cm, m].astype(float); ok = x.notna() & y.notna(); dd = (x - y)[ok]; w = A.loc[cm, 'writer_id'][ok]
                lo, hi = boot_ci(dd, w.to_numpy())
                pair.append(dict(method=meth, eval_set=es, input=inp, protocol=prot, metric=m, reference=('e0_ms8' if es.startswith('ouhdl') else 'e0_ms8_x4a'),
                    mean_m5_on_pairs=float(x[ok].mean()), mean_e0_on_pairs=float(y[ok].mean()), mean_diff_m5_minus_e0=float(dd.mean()), ci95_lo=lo, ci95_hi=hi,
                    n_pairs=int(ok.sum()), n_units=len(cm), n_writers=int(w.nunique()), ci='paired per-unit difference on units where both values are defined, writer-cluster bootstrap B=2000 seed 0'))
    pd.DataFrame(pair).to_csv(f'{O}/paired_M5_vs_E0.csv', index=False)
    st = pd.read_csv(f'{X}/report/unit_steps_eval_M5.csv'); sp = []
    src = [(f,) + meta(f) for f in rows] + [(None, k[0], k[1], ('e0_ms8' if k[0].startswith('ouhdl') else 'e0_ms8_x4a')) for k in E0R]
    for f, es, inp, meth in src:
        d = rows[f] if f else E0D[(es, inp)]; s = st[st.eval_set == es].set_index('unit_id').step; assert d.unit_id.map(s).notna().all()
        d = d.assign(step_group=np.where(d.unit_id.map(s) > 1 + 1e-6, 'step>1 (n_points_1px>320)', 'step=1'))
        for (prot, grp), g in d.groupby(['protocol', 'step_group']):
            for m in ['aiou','ldtw','dtw','kendall_tau','direction_acc','strict_order_acc','chamfer']:
                if prot == 'standardised' and m in ORDER: continue
                lo, hi = boot_ci(g[m], g.writer_id.to_numpy())
                sp.append(dict(method=meth, eval_set=es, input=inp, protocol=prot, step_group=grp, metric=m, mean=float(g[m].astype(float).mean()), ci95_lo=lo, ci95_hi=hi,
                               n_units=int(g.unit_id.nunique()), n_valid=int(g[m].notna().sum()), n_writers=int(g.writer_id.nunique())))
    pd.DataFrame(sp).to_csv(f'{O}/stepsplit_M5.csv', index=False)
    sel = []
    for arm, f in [('joint', f'{X}/runs/joint/eval/selection.json'), ('luo', f'{X}/runs/luo/eval/selection.json')] + [(f'ft_{q}', f'{X}/runs/ft/{q}/eval/selection.json') for q in ['p01','p10','p30','p100']]:
        s = json.load(open(f))
        for r in s['pooled']: sel.append(dict(arm=arm, valset='pooled', selected=(r['milestone'] == s['best_milestone']), **r))
        for r in s['per_valset']: sel.append(dict(arm=arm, selected=(r['milestone'] == s['best_milestone']), **r))
    e0s = json.load(open(f'{H}/selection.json'))
    for r in e0s['table']:
        sel.append(dict(arm='e0 (reference, 1,000 OUHD-L val units)', valset='ouhdl_val1000', milestone=int(r['method'].replace('e0_ms','')), selected=(r['method'] == e0s['best']),
                        ldtw=r['ldtw'], aiou=r['aiou'], dtw=r['dtw'], direction_acc=r['direction_acc']))
    pd.DataFrame(sel).sort_values(['arm','valset','milestone']).to_csv(f'{O}/selection_M5.csv', index=False)
    K = al[(al.protocol == 'native') & al.metric.isin(['aiou','ldtw','dtw','kendall_tau','direction_acc','strict_order_acc'])].copy()
    K['v'] = K.apply(lambda r: f"{r['mean']:.4f} [{r.ci95_lo:.4f}, {r.ci95_hi:.4f}]", axis=1)
    piv = K.pivot_table(index=['eval_set','input','label'], columns='metric', values='v', aggfunc='first')
    piv.join(al[(al.protocol == 'native') & (al.metric == 'aiou')].set_index(['eval_set','input','label'])[['n_units','n_writers']]).to_csv(f'{O}/keytable_native_M5.csv')
if __name__ == '__main__':
    main()
