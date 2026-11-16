"""OUHD-L ligature division, version 3: per-line, transcript-aligned (14-15 Oct 2026).

Each line is solved on its own, from its own pen-down strokes (in recording order) and its own
transcript.  No unit is decided by a global threshold alone.

1. Stroke classes.  Every pen-down stroke is a *dot* (both bbox sides < 0.18 x 56 px), a
   *secondary* stroke (delayed and small: path < 30 px and height < 12 px; or delayed and a
   near-straight bar: chord/path > 0.9, path < 40 px, height < 20 px; never a straight steep
   stroke with height > 2 x width), or a *body* stroke.  'Delayed' = it x-overlaps (>= 50 % of
   its width) an earlier-started stroke that extends below it, or an earlier-started stroke lies
   to its left.  Same rules as v2, applied per stroke instead of per touching component.
2. Text ligatures.  The transcript is split into ligatures after Urdu non-joining letters
   (ا آ أ د ڈ ذ ر ڑ ز ژ و ؤ ے ۓ ء ۂ ۃ); digits and punctuation are separate.
3. Alignment.  Dynamic programming over (body strokes in time order) x (text ligatures).  A
   group of 1..R consecutive body strokes is aligned to 1..MJ consecutive ligatures of one word.
   Cost of a group = gap cost (each later stroke's nearest distance to the group's earlier
   strokes, beyond 1 px, / 5 px) + length cost lam * log((L+5)/(E+5))^2, where L is the group's
   ink path length and E the expected length (sum of per-letter lengths x a per-line scale) +
   touch penalty when the group's first stroke touches the previous body stroke + Sj per extra
   joined ligature + Sm if a multi-stroke group spans several ligatures.  Skipping a text
   ligature costs St, leaving a body stroke unaligned costs Si (it is then attached as a mark).
   Touching ligatures are split whenever they are separate strokes.  Optional ``gmax`` (off by
   default) forbids a group whose later stroke is more than gmax px from the group's earlier strokes;
   On the 160 judged regions (Sj = 1.0), gmax = None / 8 / 5 / 3 / 2 px matched judge 1's winner in
   58 / 60 / 59 / 62 / 62 of 130 regions and judge 2's winner in 102 / 101 / 95 / 91 / 90 of 145, so it
   slightly raises agreement with judge 1 and lowers it with judge 2; it is off by default.
4. Marks (dots, secondary strokes, unaligned strokes) go to the group minimising
   2*dx + 0.5*d_nearest (px) + timing cost (0 for the group most recently started before the
   mark; wsame for an earlier group of the same word; wother otherwise) + wexp if the group's
   text has no dotted or marked letter.
5. Per-letter lengths are fitted on the TRAIN split only (EM: solve, then non-negative least
   squares of group length on letter counts, normalised per line; letters with < 30
   occurrences keep 1.0).

CLI:  python ligseg_v3.py fit   --root DATA --out letter_lengths.json
      python ligseg_v3.py apply --root DATA --split test --lengths letter_lengths.json --out DIR
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, unicodedata
from multiprocessing import get_context
import numpy as np
from scipy.spatial import cKDTree

from uhwr_baselines import ouhdl as O

REF = O.LINE_H - 2 * O.LINE_MARGIN  # 56 px
NONJOIN = set('اآأدڈذرڑزژوؤےۓءۂۃ')
LET_IGNORE = set('\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670\u0654\u0655\u0656\u200c\u200d')
DOTTED = set('بپتٹثجچخذڈزژڑشضظغفقنیئيةۓؤآأ') | {'\u064b', '\u064c', '\u064d', '\u064e', '\u064f', '\u0650',
                                                '\u0651', '\u0652', '\u0670', '\u0654', '\u0655'}
KAFS = set('کگك')
PARAMS = dict(wc=1.0, lam=1.0, wt=1.5, St=4.0, Si=3.0, R=6, Sj=1.0, MJ=3, Sm=4.0,
              wsame=3.0, wother=10.0, wexp=8.0, gmax=None)


# ---------------------------------------------------------------- text
def text_ligs(t):
    out = []
    for w in t.split():
        cur = ''
        for ch in w:
            if unicodedata.category(ch).startswith('M') or ch in '\u200c\u200d':
                cur += ch; continue
            if ch.isdigit() or unicodedata.category(ch).startswith('P') or ch in '۔،؟':
                if cur: out.append(cur); cur = ''
                out.append(ch); continue
            cur += ch
            if ch in NONJOIN: out.append(cur); cur = ''
        if cur: out.append(cur)
    return out


def text_ligs_w(t):
    return [(l, wi) for wi, w in enumerate(t.split()) for l in text_ligs(w)]


def letters_of(lig):
    return [ch for ch in lig if ch not in LET_IGNORE] or [lig[:1]]


def expect_mark(lig):
    """1 = must have a mark, 0 = must not, None = ambiguous."""
    if any(ch in DOTTED for ch in lig.rstrip('ی')) or ('ی' in lig[:-1]):
        return 1
    if any(ch in KAFS for ch in lig) or lig.endswith('ی') or lig in ('۔', '،'):
        return None
    return 0


# ---------------------------------------------------------------- strokes
def stroke_feats(ink):
    F = []
    for i, s in enumerate(ink):
        x0, y0 = s.min(0); x1, y1 = s.max(0)
        L = float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum()) if len(s) > 1 else 0.0
        ch = float(np.linalg.norm(s[-1] - s[0])) if len(s) > 1 else 0.0
        F.append(dict(i=i, x0=x0, y0=y0, x1=x1, y1=y1, w=x1 - x0, h=y1 - y0, L=L, straight=ch / max(L, 1e-6), P=s))
    for q in F:
        q['dot'] = q['w'] < O.DOT_FRAC * REF and q['h'] < O.DOT_FRAC * REF
    big = [q for q in F if not q['dot']]
    for q in F:
        ov = [m for m in big if m is not q and min(q['x1'], m['x1']) - max(q['x0'], m['x0']) >= 0.5 * max(q['w'], 1)
              and m['y1'] > q['y1'] + 2]
        q['delayed'] = any(m['i'] < q['i'] for m in ov) or \
            any(m['x1'] < q['x0'] + 0.2 * q['w'] and m['i'] < q['i'] for m in big if m is not q)
        steep = q['straight'] > 0.9 and q['h'] > 2.0 * max(q['w'], 1e-6)
        small = q['L'] < 30 and q['h'] < 12
        bar = q['straight'] > 0.9 and q['L'] < 40 and q['h'] < 20
        q['kind'] = 'dot' if q['dot'] else ('sec' if (q['delayed'] and not steep and (small or bar)) else 'body')
    return F


# ---------------------------------------------------------------- alignment
def solve(F, t, LEN=None, W=None):
    W = {**PARAMS, **(W or {})}
    TL = text_ligs_w(t); N = len(TL)
    B = [q for q in F if q['kind'] == 'body']; K = len(B)
    ew = [sum((LEN.get(ch, 1.0) if LEN else 1.0) for ch in letters_of(l)) for l, _ in TL]
    scale = (sum(q['L'] for q in B) + 1e-6) / max(sum(ew), 1e-6)
    trees = [cKDTree(q['P']) for q in B]
    D = {}

    def d(a, b):
        if (a, b) not in D:
            D[(a, b)] = float(trees[b].query(B[a]['P'])[0].min())
        return D[(a, b)]
    R = W['R']; conn = {}
    for a in range(K):
        acc = 0.0; conn[(a, a + 1)] = 0.0
        for b in range(a + 1, min(K, a + R)):
            g = min(d(b, x) for x in range(a, b))
            if W['gmax'] is not None and g > W['gmax']:
                break  # a later stroke beyond gmax px from the group cannot belong to it
            acc += W['wc'] * max(0.0, g - 1.0) / 5.0
            conn[(a, b + 1)] = acc
    btouch = [0.0] + [W['wt'] if d(k, k - 1) < 1.5 else 0.0 for k in range(1, K)]
    cumL = np.r_[0, np.cumsum([q['L'] for q in B])]; cume = np.r_[0, np.cumsum(ew)]
    INF = 1e18; dp = np.full((K + 1, N + 1), INF); bp = {}; dp[0, 0] = 0
    for k in range(K + 1):
        for j in range(N + 1):
            c = dp[k, j]
            if c >= INF: continue
            if j < N and c + W['St'] < dp[k, j + 1]: dp[k, j + 1] = c + W['St']; bp[(k, j + 1)] = ('skipT', k, j)
            if k < K and c + W['Si'] < dp[k + 1, j]: dp[k + 1, j] = c + W['Si']; bp[(k + 1, j)] = ('extra', k, j)
            for m in range(1, min(W['MJ'], N - j) + 1):
                if m > 1 and TL[j + m - 1][1] != TL[j][1]: break
                exp = scale * (cume[j + m] - cume[j])
                for r in range(1, min(R, K - k) + 1):
                    if (k, k + r) not in conn: break
                    Lg = cumL[k + r] - cumL[k]
                    cc = (c + conn[(k, k + r)] + W['lam'] * np.log((Lg + 5) / (exp + 5)) ** 2 + (btouch[k] if k > 0 else 0)
                          + W['Sj'] * (m - 1) + (W['Sm'] if (m > 1 and r > 1) else 0.0))
                    if cc < dp[k + r, j + m]: dp[k + r, j + m] = cc; bp[(k + r, j + m)] = ('grp', k, j, r, m)
    k, j = K, N; groups, extras, skipped = [], [], []
    while (k, j) != (0, 0):
        op = bp[(k, j)]
        if op[0] == 'grp':
            groups.append(dict(ligs=list(range(op[2], op[2] + op[4])), strokes=[B[x]['i'] for x in range(op[1], op[1] + op[3])],
                               L=float(cumL[op[1] + op[3]] - cumL[op[1]])))
            k, j = op[1], op[2]
        elif op[0] == 'extra': extras.append(B[op[1]]['i']); k, j = op[1], op[2]
        else: skipped.append(op[2]); k, j = op[1], op[2]
    return dict(groups=groups[::-1], extras=extras, skipped=sorted(skipped), TL=TL, cost=float(dp[K, N]), scale=scale)


def group_expect(TL, g):
    es = [expect_mark(TL[j][0]) for j in g['ligs']]
    if any(e == 1 for e in es): return 1
    if all(e == 0 for e in es): return 0
    return None


def attach(F, ink, s, W=None):
    W = {**PARAMS, **(W or {})}
    G = s['groups']
    if not G:  # degenerate: one unit
        return [dict(ligs=[], strokes=sorted(q['i'] for q in F), marks=[], t0=0)]
    for g in G:
        g['P'] = np.vstack([ink[i] for i in g['strokes']]); g['x0'], g['y0'] = g['P'].min(0); g['x1'], g['y1'] = g['P'].max(0)
        g['t0'] = min(g['strokes']); g['words'] = {s['TL'][j][1] for j in g['ligs']}
        g['exp'] = group_expect(s['TL'], g); g['marks'] = []; g['tree'] = cKDTree(g['P'])
    marks = [q for q in F if q['kind'] != 'body'] + [F[i] for i in s['extras']]
    for m in marks:
        xc = 0.5 * (m['x0'] + m['x1'])
        prev = [gi for gi, g in enumerate(G) if g['t0'] < m['i']]; gp = prev[-1] if prev else 0
        best = None
        for gi, g in enumerate(G):
            dx = max(0.0, g['x0'] - xc, xc - g['x1']); dn = float(g['tree'].query(m['P'])[0].min())
            c = 2 * dx + 0.5 * dn
            if gi != gp: c += W['wsame'] if (g['words'] & G[gp]['words'] and gi < gp) else W['wother']
            if g['exp'] == 0: c += W['wexp']
            if best is None or c < best[0]: best = (c, gi)
        G[best[1]]['marks'].append(m['i'])
    return G


def divide_line(ink, t, LEN=None, W=None):
    F = stroke_feats(ink)
    s = solve(F, t, LEN, W)
    G = attach(F, ink, s, W)
    units = []
    for g in G:
        units.append(dict(stroke_idx=sorted(g['strokes'] + g['marks']), body_stroke_idx=sorted(g['strokes']),
                          mark_stroke_idx=sorted(g['marks']), text='+'.join(s['TL'][j][0] for j in g['ligs']),
                          n_text_ligs=len(g['ligs']), t0=int(min(g['strokes'])) if g['strokes'] else 0))
    units.sort(key=lambda u: u['t0'])  # writing order of the bodies
    kinds = {q['i']: q['kind'] for q in F}
    return units, dict(skipped=[s['TL'][j][0] for j in s['skipped']], n_text=len(s['TL']), extras=s['extras'],
                       kinds=kinds, groups_L=[(g['ligs'], g['L']) for g in s['groups']], scale=s['scale'])


# ---------------------------------------------------------------- IO / CLI
def _load(root, split):
    man = O.load_manifest(root, split)
    out = []
    for _, r in man.iterrows():
        out.append((r.sample_id, os.path.join(root, r.csv), r.line))
    return out


def _work(args):
    sid, csv, t, LEN, W = args
    ln = O.make_line(sid, O.load_raw_strokes(csv))
    units, info = divide_line(ln.ink, t, LEN, W)
    return sid, units, info


def run_split(root, split, LEN=None, W=None, procs=12):
    items = [(sid, csv, t, LEN, W) for sid, csv, t in _load(root, split)]
    with get_context('fork').Pool(procs) as pool:
        res = pool.map(_work, items, chunksize=8)
    return {sid: (u, i) for sid, u, i in res}, {sid: t for sid, _, t in _load(root, split)}


def fit_lengths(root, iters=4, min_count=30, procs=12):
    from scipy.optimize import nnls
    LEN = None; log = []
    for it in range(iters):
        res, texts = run_split(root, 'train', LEN, None, procs)
        chars = sorted({ch for t in texts.values() for l in text_ligs(t) for ch in letters_of(l)})
        ci = {c: i for i, c in enumerate(chars)}
        A, y = [], []
        for sid, (units, info) in res.items():
            TL = text_ligs_w(texts[sid])
            for ligs, L in info['groups_L']:
                if len(ligs) != 1: continue
                row = np.zeros(len(chars))
                for ch in letters_of(TL[ligs[0]][0]): row[ci[ch]] += 1
                A.append(row); y.append(L / info['scale'])
        A = np.array(A); y = np.array(y)
        coef, _ = nnls(A, y); cnt = (A > 0).sum(0)
        new = {c: (float(coef[i]) if cnt[i] >= min_count and coef[i] > 0.05 else 1.0) for c, i in ci.items()}
        mean = np.mean([new[c] for c in chars if cnt[ci[c]] >= min_count]); new = {c: v / mean for c, v in new.items()}
        delta = None if LEN is None else max(abs(new[c] - LEN.get(c, 1.0)) for c in chars)
        log.append(dict(iter=it, n_groups=int(len(y)), max_change=delta))
        print(log[-1], flush=True)
        LEN = new
    counts = {c: int(cnt[ci[c]]) for c in chars}
    return LEN, counts, log


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['fit', 'apply'])
    ap.add_argument('--root', required=True); ap.add_argument('--split', default='test')
    ap.add_argument('--lengths'); ap.add_argument('--out', required=True); ap.add_argument('--procs', type=int, default=12)
    a = ap.parse_args(); t0 = time.time()
    if a.cmd == 'fit':
        LEN, counts, log = fit_lengths(a.root, procs=a.procs)
        json.dump(dict(lengths=LEN, train_counts=counts, em_log=log, params=PARAMS,
                       code_sha256=sha256(__file__), train_manifest_sha256=sha256(os.path.join(a.root, 'train.csv')),
                       seconds=time.time() - t0), open(a.out, 'w'), ensure_ascii=False, indent=1)
    else:
        LEN = json.load(open(a.lengths))['lengths'] if a.lengths else None
        res, texts = run_split(a.root, a.split, LEN, None, a.procs)
        os.makedirs(a.out, exist_ok=True)
        json.dump({sid: dict(units=u, skipped=i['skipped'], n_text=i['n_text'], extras=i['extras'],
                             kinds={int(k): v for k, v in i['kinds'].items()}) for sid, (u, i) in res.items()},
                  open(os.path.join(a.out, f'units_v3_{a.split}.json'), 'w'), ensure_ascii=False)
        print('lines', len(res), 'units', sum(len(u) for u, _ in res.values()), 'seconds', round(time.time() - t0, 1))
