"""router336 (SPEC_router336.md): ScriptCNN with the Urdu class = 336 p01 units only.
arm natural : script_clf.main loop verbatim on 80,000 non-Urdu rows of m5_joint_train + 336 p01 rows.
arm balanced: Line R day-1 loop (rr_train.py): 80,000 non-Urdu rows + 20,000 Urdu draws with replacement per epoch.
Saves best-val checkpoint (script_clf.pt, recipe rule) and last-epoch checkpoint (script_clf_last.pt). CPU only."""
import argparse, json, os, time, sys, hashlib
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from script_clf import ScriptCNN, prep, predict, rtl_prob, SCRIPTS, RTL
ap = argparse.ArgumentParser()
ap.add_argument('--pull', required=True); ap.add_argument('--e0train', required=True); ap.add_argument('--ids', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--arm', required=True, choices=['natural', 'balanced'])
ap.add_argument('--epochs', type=int, default=8); ap.add_argument('--bs', type=int, default=256)
ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--patience', type=int, default=8); ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--urdu_per_epoch', type=int, default=20000)
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(a.threads)
os.makedirs(a.out, exist_ok=False); t0 = time.time()
zt = np.load(os.path.join(a.pull, 'train_m5_joint_cond.npz'), allow_pickle=True)
st = zt['scripts'].astype(str); keep = st != 'ouhdl'
Xl = zt['cond_rendered'][keep]; yl = np.array([SCRIPTS.index(s) for s in st[keep]])
ids = [l.strip() for l in open(a.ids) if l.strip()]
ze = np.load(a.e0train, allow_pickle=True); eu = ze['unit_ids'].astype(str)
pos = {u: i for i, u in enumerate(eu)}; sel = np.array([pos[u] for u in ids])
Xu = ze['cond_rendered'][sel]
info = {'train_counts_unique': {s: int((np.concatenate([yl, np.zeros(len(Xu), int)]) == i).sum()) for i, s in enumerate(SCRIPTS)},
        'joint_train_urdu_rows_dropped': int((~keep).sum()), 'p01_units': len(ids),
        'p01_unit_ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
        'p01_images_sha256': hashlib.sha256(Xu.tobytes()).hexdigest(),
        'p01_in_joint_train': int(np.isin(ids, zt['unit_ids'].astype(str)).sum())}
assert info['p01_units'] == 336 and info['p01_in_joint_train'] == 0
assert info['p01_images_sha256'] == '2ddb21e9b93cea1dcc299770ab1684d8c8271814cc8c53653690e22aeeade066'
del zt, ze
Xtr = np.concatenate([Xl, Xu]); ytr = np.concatenate([yl, np.zeros(len(Xu), int)]); nl, nu = len(Xl), len(Xu)
Xva, yva = [], []
for s in SCRIPTS:
    z = np.load(os.path.join(a.pull, f'val_{s}_cond.npz'), allow_pickle=True)
    Xva.append(z['cond_rendered']); yva.append(np.full(len(z['cond_rendered']), SCRIPTS.index(s)))
Xva = np.concatenate(Xva); yva = np.concatenate(yva)
print('arm', a.arm, 'seed', a.seed, info, 'val', Xva.shape, np.bincount(yva), flush=True)
model = ScriptCNN(); n_par = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
n_epoch = len(Xtr) if a.arm == 'natural' else nl + a.urdu_per_epoch
steps_per_epoch = int(np.ceil(n_epoch / a.bs))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch)
def ck(ep, acc, tag):
    return {'state_dict': model.state_dict(), 'scripts': SCRIPTS, 'rtl': sorted(RTL), 'epoch': ep, 'val_acc5': acc,
            'arch': 'ScriptCNN w=(16,32,64,96)', 'n_params': n_par, 'seed': a.seed, 'arm': a.arm, 'ckpt_rule': tag,
            'urdu_train': 'p01 (336 units) only'}
best, best_ep, bad, log = -1, -1, 0, []
for ep in range(a.epochs):
    model.train()
    if a.arm == 'natural':
        perm = np.random.permutation(len(Xtr))
    else:
        ep_idx = np.concatenate([np.arange(nl), nl + np.random.randint(0, nu, a.urdu_per_epoch)])
        perm = ep_idx[np.random.permutation(len(ep_idx))]
    t = time.time(); tl = 0.0
    for i in range(0, len(perm), a.bs):
        idx = perm[i:i+a.bs]
        x = prep(Xtr[idx]); y = torch.from_numpy(ytr[idx])
        dx, dy = np.random.randint(-3, 4, 2)
        x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(2, 3))
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item() * len(idx)
    pva = predict(model, Xva); acc = float((pva.argmax(1) == yva).mean())
    acc_rtl = float(((rtl_prob(pva).argmax(1) == 0) == np.isin(yva, [SCRIPTS.index(s) for s in RTL])).mean())
    per = {s: float((pva.argmax(1)[yva == i] == i).mean()) for i, s in enumerate(SCRIPTS)}
    log.append({'epoch': ep, 'train_loss': tl / len(perm), 'val_acc5': acc, 'val_acc_rtl': acc_rtl, 'val_acc_per_script': per,
                'urdu_rows_in_epoch': int((perm >= nl).sum()), 'sec': time.time() - t})
    print(log[-1], flush=True)
    if acc > best:
        best, best_ep, bad = acc, ep, 0
        torch.save(ck(ep, acc, 'best_val_acc5'), os.path.join(a.out, 'script_clf.pt'))
    else:
        bad += 1
        if bad >= a.patience: print('early stop', flush=True); break
torch.save(ck(ep, acc, 'last_epoch'), os.path.join(a.out, 'script_clf_last.pt'))
json.dump({'log': log, 'best_epoch': best_ep, 'best_val_acc5': best, 'last_epoch': ep, 'n_params': n_par, 'args': vars(a),
           'data_info': info, 'steps_per_epoch': steps_per_epoch, 'torch': torch.__version__, 'numpy': np.__version__,
           'wall_s': time.time() - t0, 'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},
          open(os.path.join(a.out, 'train_log.json'), 'w'), indent=1)
print('done', time.time() - t0, flush=True)
