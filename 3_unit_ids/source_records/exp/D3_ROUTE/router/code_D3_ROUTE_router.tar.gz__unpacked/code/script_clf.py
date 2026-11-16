"""Script-family classifier for routing (D3_ROUTE router).
Small CNN on the 64x64 E0 condition image (cond_rendered, uint8, black ink on white).
Outputs: P(script in {ouhdl, casia_zh, casia_en, hpl_tamil, iam}) and collapsed P(RTL, LTR) (ouhdl = RTL).
Train on m5_joint_train (100k, 20k per script), early stopping on the m5 val sets (5 x 250).
CPU only. Single seed.
"""
import argparse, json, os, time, hashlib
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

SCRIPTS = ['ouhdl', 'casia_zh', 'casia_en', 'hpl_tamil', 'iam']
RTL = {'ouhdl'}

class ScriptCNN(nn.Module):
    def __init__(self, n_cls=5, w=(16, 32, 64, 96)):
        super().__init__()
        chans = [1] + list(w)
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(chans[i], chans[i+1], 3, padding=1), nn.BatchNorm2d(chans[i+1]), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)       # 64 -> 4
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(chans[-1]*4*4, 128), nn.ReLU(inplace=True), nn.Linear(128, n_cls))
    def forward(self, x):
        return self.head(self.features(x).flatten(1))

def prep(imgs):
    # uint8 black-on-white -> float ink=1, background=0
    x = torch.from_numpy(imgs).float().div_(255.0)
    return (1.0 - x).unsqueeze(1)

def rtl_prob(p5):
    idx = [SCRIPTS.index(s) for s in SCRIPTS if s in RTL]
    p_rtl = p5[:, idx].sum(1)
    return np.stack([p_rtl, 1 - p_rtl], 1)

@torch.no_grad()
def predict(model, imgs, bs=1024):
    model.eval(); out = []
    for i in range(0, len(imgs), bs):
        out.append(F.softmax(model(prep(imgs[i:i+bs])), 1).numpy())
    return np.concatenate(out) if out else np.zeros((0, len(SCRIPTS)))

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pull', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--threads', type=int, default=12)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    tr = np.load(os.path.join(a.pull, 'train_m5_joint_cond.npz'), allow_pickle=True)
    Xtr = tr['cond_rendered']; str_tr = tr['scripts'].astype(str)
    ytr = np.array([SCRIPTS.index(s) for s in str_tr])
    Xva, yva = [], []
    for s in SCRIPTS:
        z = np.load(os.path.join(a.pull, f'val_{s}_cond.npz'), allow_pickle=True)
        Xva.append(z['cond_rendered']); yva.append(np.full(len(z['cond_rendered']), SCRIPTS.index(s)))
    Xva = np.concatenate(Xva); yva = np.concatenate(yva)
    print('train', Xtr.shape, np.bincount(ytr), 'val', Xva.shape, np.bincount(yva), flush=True)
    model = ScriptCNN()
    n_par = sum(p.numel() for p in model.parameters()); print('params', n_par, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_per_epoch = int(np.ceil(len(Xtr) / a.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch)
    best, best_ep, bad, log = -1, -1, 0, []
    for ep in range(a.epochs):
        model.train(); perm = np.random.permutation(len(Xtr)); t = time.time(); tl = 0.0
        for i in range(0, len(Xtr), a.bs):
            idx = perm[i:i+a.bs]
            x = prep(Xtr[idx]); y = torch.from_numpy(ytr[idx])
            # light augmentation: random shift up to 3 px
            dx, dy = np.random.randint(-3, 4, 2)
            x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(2, 3))
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item() * len(idx)
        pva = predict(model, Xva); acc = float((pva.argmax(1) == yva).mean())
        acc_rtl = float(((rtl_prob(pva).argmax(1) == 0) == np.isin(yva, [SCRIPTS.index(s) for s in RTL])).mean())
        log.append({'epoch': ep, 'train_loss': tl / len(Xtr), 'val_acc5': acc, 'val_acc_rtl': acc_rtl, 'sec': time.time() - t})
        print(log[-1], flush=True)
        if acc > best:
            best, best_ep, bad = acc, ep, 0
            torch.save({'state_dict': model.state_dict(), 'scripts': SCRIPTS, 'rtl': sorted(RTL), 'epoch': ep, 'val_acc5': acc, 'arch': 'ScriptCNN w=(16,32,64,96)', 'n_params': n_par, 'seed': a.seed}, os.path.join(a.out, 'script_clf.pt'))
        else:
            bad += 1
            if bad >= a.patience: print('early stop', flush=True); break
    json.dump({'log': log, 'best_epoch': best_ep, 'best_val_acc5': best, 'n_params': n_par, 'args': vars(a),
               'torch': torch.__version__, 'numpy': np.__version__}, open(os.path.join(a.out, 'train_log.json'), 'w'), indent=1)

if __name__ == '__main__':
    main()
