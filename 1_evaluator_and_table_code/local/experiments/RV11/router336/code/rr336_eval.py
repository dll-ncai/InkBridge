"""router336 evaluation (SPEC_router336.md section 7). Decision: routed to the Urdu model iff P(ouhdl) >= 0.5.
--check-only: re-score the reported checkpoint and compare with the stored D3_ROUTE manifests, then exit."""
import argparse, json, os, sys, hashlib, gzip
import numpy as np, pandas as pd, torch
from scipy.stats import binomtest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from script_clf import ScriptCNN, predict, SCRIPTS, sha256
ap = argparse.ArgumentParser(); ap.add_argument('--check-only', action='store_true'); a = ap.parse_args()
torch.set_num_threads(2)
R = '$LOCAL/experiments/RV11/router336'; D3 = '$LOCAL/experiments/D3_ROUTE/router'
SETS = [('test_ouhdl', 'ouhdl'), ('test_casia_zh', 'casia_zh'), ('test_casia_en', 'casia_en'), ('test_hpl_tamil', 'hpl_tamil'),
        ('test_iam500', 'iam'), ('german_unipen_de_test', None), ('german_onhw_all', None)]
def wilson(k, n, z=1.959963984540054):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return c - h, c + h
def load(path):
    m = ScriptCNN(); ck = torch.load(path, map_location='cpu', weights_only=False); m.load_state_dict(ck['state_dict']); m.eval()
    assert ck['scripts'] == SCRIPTS; return m, ck
data = {}
for s, true in SETS:
    z = np.load(f'{R}/pull/{s}_cond.npz', allow_pickle=True); man = pd.read_csv(f'{D3}/manifest_{s}.csv')
    u = z['unit_ids'].astype(str); assert len(u) == len(man) and (man['unit_id'].astype(str).values == u).all(), s
    data[s] = (z['cond_rendered'], u, man, true)
models = [('reported', 'best', 0, f'{D3}/script_clf.pt')]
if not a.check_only:
    models.append(('lineR_day1_balanced_cluster', 'best', 0, f'{R}/lineR_ref/run_p01_s0/script_clf_p01.pt'))
    for arm in ['natural', 'balanced']:
        for seed in [0, 1, 2]:
            models.append((arm, 'best', seed, f'{R}/runs/{arm}_s{seed}/script_clf.pt'))
            models.append((arm, 'last', seed, f'{R}/runs/{arm}_s{seed}/script_clf_last.pt'))
rows, units, shas, ref = [], [], [], {}
for arm, rule, seed, path in models:
    m, ck = load(path); shas.append(dict(arm=arm, ckpt_rule=rule, seed=seed, path=os.path.relpath(path, '$LOCAL'), sha256=sha256(path), epoch=ck.get('epoch'), val_acc5=ck.get('val_acc5')))
    for s, true in SETS:
        X, u, man, _ = data[s]; p = predict(m, X); routed = p[:, 0] >= 0.5; n = len(u); k = int(routed.sum())
        if arm == 'reported':
            ref[s] = routed; mr = man['p_ouhdl'].values >= 0.5
            chk = dict(set=s, n=n, rescored_n_routed=k, manifest_n_routed=int(mr.sum()), unit_mismatch=int((mr != routed).sum()),
                       max_abs_dp=float(np.abs(man['p_ouhdl'].values - p[:, 0]).max()), manifest_pred_script_mismatch=int((man['pred_script'].values != np.array(SCRIPTS)[p.argmax(1)]).sum()))
            print(chk, flush=True); assert chk['unit_mismatch'] == 0, 'pipeline check failed'
        correct = k if true == 'ouhdl' else n - k; lo, hi = wilson(correct, n)
        gained = int((routed & ~ref[s]).sum()); lost = int((~routed & ref[s]).sum())
        rows.append(dict(arm=arm, ckpt_rule=rule, seed=seed, set=s, n=n, n_routed_urdu=k, route_acc=correct / n, route_acc_wilson_lo=lo, route_acc_wilson_hi=hi,
                         acc_script5=(float((np.array(SCRIPTS)[p.argmax(1)] == true).mean()) if true else np.nan),
                         n_route_gained_vs_reported=gained, n_route_lost_vs_reported=lost, n_route_changed_vs_reported=gained + lost,
                         mcnemar_exact_p=(binomtest(min(gained, lost), gained + lost, 0.5).pvalue if gained + lost else 1.0), identical_to_reported=int(gained + lost == 0)))
        units.append(pd.DataFrame(dict(arm=arm, ckpt_rule=rule, seed=seed, set=s, unit_id=u, p_ouhdl=np.round(p[:, 0], 6), pred_script=np.array(SCRIPTS)[p.argmax(1)], routed_urdu=routed.astype(int))))
if a.check_only: print('PIPELINE CHECK PASSED'); sys.exit(0)
df = pd.DataFrame(rows); df.insert(0, 'row_type', 'per_seed'); df.loc[df.arm.isin(['reported', 'lineR_day1_balanced_cluster']), 'row_type'] = 'single_checkpoint'
pool = []
for (arm, rule, s), g in df[df.row_type == 'per_seed'].groupby(['arm', 'ckpt_rule', 'set'], sort=False):
    pool.append(dict(row_type='pooled_3seeds', arm=arm, ckpt_rule=rule, seed='0;1;2', set=s, n=int(g.n.iloc[0]), n_routed_urdu=g.n_routed_urdu.mean(), n_routed_urdu_min=int(g.n_routed_urdu.min()), n_routed_urdu_max=int(g.n_routed_urdu.max()),
                     route_acc=g.route_acc.mean(), route_acc_min=g.route_acc.min(), route_acc_max=g.route_acc.max(), acc_script5=g.acc_script5.mean(), acc_script5_min=g.acc_script5.min(), acc_script5_max=g.acc_script5.max(),
                     n_route_changed_vs_reported=g.n_route_changed_vs_reported.mean(), n_seeds_identical_to_reported=int(g.identical_to_reported.sum())))
out = pd.concat([df, pd.DataFrame(pool)], ignore_index=True)
rep = df[df.arm == 'reported'].set_index('set'); out['reported_n_routed_urdu'] = out.set.map(rep.n_routed_urdu); out['reported_route_acc'] = out.set.map(rep.route_acc); out['reported_acc_script5'] = out.set.map(rep.acc_script5)
out.to_csv(f'{R}/router336_results.csv', index=False)
pd.DataFrame(shas).to_csv(f'{R}/router336_checkpoints_sha256.csv', index=False)
pd.concat(units, ignore_index=True).to_csv(f'{R}/eval/router336_per_unit.csv.gz', index=False)
print('wrote', len(out), 'rows')
