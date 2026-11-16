"""Apply the script classifier to the m5 val/test sets and the German sets; write per-set routing manifests,
accuracy / confusion tables and route counts.
Specialists: ouhdl -> E0_ms8; iam, casia_en -> X5_ms9; casia_zh -> E27_ms47; hpl_tamil and 'other' -> M5_joint_ms9.
'other' = max script probability < --thr (default 0.5). Oracle for German = 'other' (unseen family) -> M5_joint_ms9.
"""
import argparse, json, os, sys, hashlib
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from script_clf import ScriptCNN, SCRIPTS, RTL, predict, rtl_prob, sha256

SPECIALIST = {'ouhdl': 'E0_ms8', 'iam': 'X5_ms9', 'casia_en': 'X5_ms9', 'casia_zh': 'E27_ms47', 'hpl_tamil': 'M5_joint_ms9', 'other': 'M5_joint_ms9'}
SPECIALIST_CKPT = {'E0_ms8': 'runs/e0/e0_v3_rendered_100k_bs512 (milestone 8)', 'X5_ms9': 'exp/X5 (milestone 9)',
                   'E27_ms47': 'runs/nagamatsu/casia_full_a100x2__03/checkpoints/checkpoint_000047.pt', 'M5_joint_ms9': 'exp/M5 joint (milestone 9 = 90k)'}
SETS = {  # name -> (true script or None for unseen)
 'val_ouhdl': 'ouhdl', 'val_casia_zh': 'casia_zh', 'val_casia_en': 'casia_en', 'val_hpl_tamil': 'hpl_tamil', 'val_iam': 'iam',
 'test_ouhdl': 'ouhdl', 'test_casia_zh': 'casia_zh', 'test_casia_en': 'casia_en', 'test_hpl_tamil': 'hpl_tamil', 'test_iam500': 'iam',
 'german_unipen_de_test': None, 'german_onhw_all': None}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pull', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--thr', type=float, default=0.5); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); torch.set_num_threads(8)
    ck = torch.load(a.ckpt, map_location='cpu'); model = ScriptCNN(); model.load_state_dict(ck['state_dict']); model.eval()
    meta = json.load(open(os.path.join(a.pull, 'pull_meta.json')))
    rows_acc, conf_all, route_counts, manifests = [], {}, [], {}
    for name, true_s in SETS.items():
        z = np.load(os.path.join(a.pull, f'{name}_cond.npz'), allow_pickle=True)
        p5 = predict(model, z['cond_rendered']); prtl = rtl_prob(p5)
        pred = np.array(SCRIPTS)[p5.argmax(1)]; pmax = p5.max(1)
        pred_route = np.where(pmax >= a.thr, pred, 'other')
        pred_spec = np.array([SPECIALIST[s] for s in pred_route])
        oracle_script = np.full(len(pred), true_s if true_s else 'other')
        oracle_spec = np.array([SPECIALIST[s] for s in oracle_script])
        df = pd.DataFrame({'set': name, 'unit_id': z['unit_ids'].astype(str), 'sample_id': z['sample_ids'].astype(str), 'writer_id': z['writer_ids'],
                           'true_script': true_s if true_s else 'german_unseen', 'pred_script': pred, 'p_max': pmax.round(4),
                           'p_rtl': prtl[:, 0].round(4), 'pred_family': np.where(prtl[:, 0] >= 0.5, 'RTL', 'LTR'),
                           'pred_route': pred_route, 'pred_specialist': pred_spec, 'oracle_specialist': oracle_spec})
        for i, s in enumerate(SCRIPTS): df[f'p_{s}'] = p5[:, i].round(4)
        df.to_csv(os.path.join(a.out, f'manifest_{name}.csv'), index=False); manifests[name] = len(df)
        conf = pd.crosstab(pd.Series(df.true_script, name='true'), pd.Series(df.pred_script, name='pred')).reindex(columns=SCRIPTS, fill_value=0)
        conf_all[name] = conf
        true_fam = ('RTL' if true_s in RTL else 'LTR') if true_s else None
        rows_acc.append({'set': name, 'n': len(df), 'true_script': true_s or 'german_unseen',
                         'acc_script': float((pred == true_s).mean()) if true_s else None,
                         'acc_family': float((df.pred_family == true_fam).mean()) if true_s else None,
                         'share_routed_other': float((pred_route == 'other').mean()),
                         'oracle_specialist_match': float((pred_spec == oracle_spec).mean()),
                         **{f'share_pred_{s}': float((pred == s).mean()) for s in SCRIPTS},
                         'share_pred_RTL': float((df.pred_family == 'RTL').mean()),
                         **{f'share_route_{v}': float((pred_spec == v).mean()) for v in sorted(set(SPECIALIST.values()))}})
        for spec_name, cnt in df.pred_specialist.value_counts().items():
            route_counts.append({'set': name, 'specialist': spec_name, 'n_pred': int(cnt), 'n_oracle': int((oracle_spec == spec_name).sum())})
    acc = pd.DataFrame(rows_acc); acc.to_csv(os.path.join(a.out, 'accuracy_per_set.csv'), index=False)
    pd.DataFrame(route_counts).to_csv(os.path.join(a.out, 'route_counts.csv'), index=False)
    # pooled confusion for val and for test
    for split in ('val', 'test'):
        pooled = sum(conf_all[k].reindex(index=SCRIPTS, fill_value=0) for k in conf_all if k.startswith(split))
        pooled.to_csv(os.path.join(a.out, f'confusion_{split}_pooled.csv'))
        print(split, 'pooled confusion\n', pooled)
    ger = pd.concat([conf_all[k] for k in conf_all if k.startswith('german')], keys=[k for k in conf_all if k.startswith('german')])
    ger.to_csv(os.path.join(a.out, 'german_routing_counts.csv'))
    print(acc[['set', 'n', 'acc_script', 'acc_family', 'share_routed_other', 'share_pred_RTL']].to_string())
    print(pd.DataFrame(route_counts).to_string())
    json.dump({'ckpt': a.ckpt, 'ckpt_sha256': sha256(a.ckpt), 'thr_other': a.thr, 'specialists': SPECIALIST, 'specialist_ckpts': SPECIALIST_CKPT,
               'manifests': manifests, 'sources': {k: {'source': meta[k]['source'], 'source_sha256': meta[k]['source_sha256']} for k in SETS}},
              open(os.path.join(a.out, 'routing_meta.json'), 'w'), indent=1)

if __name__ == '__main__':
    main()
