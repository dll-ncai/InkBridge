"""D1 token channel: one uint8 per unit. RTL (Urdu / ouhdl) = 255, LTR (casia_zh, casia_en, hpl_tamil, iam) = 0, unknown (German, unseen) = 128.
Reads unit_ids / sample_ids / scripts from an E0-format npz (or a cond-only extract with the same keys) and writes
<out>.npz {token uint8 (N,), unit_ids, sample_ids, script (N,) str} + <out>.json (counts, source sha256, checks).
The token order is the unit order of the source npz. Verification: the `scripts` key (when present) and the
'<script>|' prefix of unit_ids (when present) must agree with the script assignment; ouhdl ids without prefix must match
the OUHD-L pattern dddd_dddd_d_[LG]ddd (L = line units, G = gold units). Any disagreement raises.
"""
import argparse, json, os, re, hashlib, sys
import numpy as np

RTL_SCRIPTS = {'ouhdl'}
LTR_SCRIPTS = {'casia_zh', 'casia_en', 'hpl_tamil', 'iam'}
TOKEN = {'RTL': 255, 'LTR': 0, 'UNKNOWN': 128}
OUHDL_ID = re.compile(r'^\d{4}_\d{4}_\d+_[LG]\d{3}$')

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()

def family(s):
    if s in RTL_SCRIPTS: return 'RTL'
    if s in LTR_SCRIPTS: return 'LTR'
    return 'UNKNOWN'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True, help='output prefix (without extension)')
    ap.add_argument('--script', required=True, help="script tag, 'auto' (use the scripts key) or 'unknown' (German)")
    ap.add_argument('--set-name', required=True)
    a = ap.parse_args()
    z = np.load(a.src, allow_pickle=True)
    n = int(z['cond_rendered'].shape[0]) if 'cond_rendered' in z.files else int(len(z['unit_ids']))
    unit_ids = z['unit_ids'].astype(str) if 'unit_ids' in z.files else np.array([f'u{i:06d}' for i in range(n)])
    sample_ids = z['sample_ids'].astype(str) if 'sample_ids' in z.files else np.array([''] * n)
    checks = {}
    if a.script == 'auto':
        assert 'scripts' in z.files, 'auto needs a scripts key'
        script = z['scripts'].astype(str)
    else:
        script = np.full(n, a.script, dtype='<U16')
    # check 1: scripts key agreement
    if 'scripts' in z.files:
        checks['scripts_key_agrees'] = bool(np.array_equal(z['scripts'].astype(str), script))
        assert checks['scripts_key_agrees'], 'scripts key disagrees with assignment'
    if 'set_ids' in z.files:
        checks['set_ids_agree'] = bool(np.array_equal(z['set_ids'].astype(str), script))
        assert checks['set_ids_agree'], 'set_ids disagree with assignment'
    # check 2: unit id prefix agreement
    pref = np.array([u.split('|')[0] if '|' in u else '' for u in unit_ids])
    has_pref = pref != ''
    if has_pref.any():
        checks['unit_id_prefix_agrees'] = bool(np.array_equal(pref[has_pref], script[has_pref]))
        assert checks['unit_id_prefix_agrees'], 'unit id prefix disagrees with assignment'
    checks['n_with_prefix'] = int(has_pref.sum())
    if a.script == 'ouhdl' and not has_pref.all():
        checks['ouhdl_id_pattern_all'] = bool(all(OUHDL_ID.match(u) for u in unit_ids[~has_pref]))
        assert checks['ouhdl_id_pattern_all'], 'ouhdl ids do not match pattern'
    if a.script == 'unknown':
        checks['no_known_prefix'] = not bool(np.isin(pref, list(RTL_SCRIPTS | LTR_SCRIPTS)).any())
        assert checks['no_known_prefix']
    fam = np.array([family(s) for s in script])
    token = np.array([TOKEN[f] for f in fam], dtype=np.uint8)
    assert len(token) == n and token.dtype == np.uint8
    assert np.array_equal(token == 255, np.isin(script, list(RTL_SCRIPTS)))
    assert np.array_equal(token == 0, np.isin(script, list(LTR_SCRIPTS)))
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    np.savez(a.out + '.npz', token=token, unit_ids=unit_ids, sample_ids=sample_ids, script=script, family=fam)
    counts = {int(k): int(v) for k, v in zip(*np.unique(token, return_counts=True))}
    per_script = {str(k): int(v) for k, v in zip(*np.unique(script, return_counts=True))}
    info = {'set_name': a.set_name, 'source': os.path.abspath(a.src), 'source_sha256': sha256(a.src), 'n_units': n,
            'token_counts': counts, 'per_script': per_script, 'token_map': TOKEN, 'rtl_scripts': sorted(RTL_SCRIPTS), 'ltr_scripts': sorted(LTR_SCRIPTS),
            'checks': checks, 'token_sha256': hashlib.sha256(token.tobytes()).hexdigest(), 'unit_ids_sha256': hashlib.sha256('\n'.join(unit_ids).encode()).hexdigest(),
            'out_npz': os.path.abspath(a.out + '.npz'), 'out_npz_sha256': None, 'code_sha256': sha256(os.path.abspath(__file__))}
    info['out_npz_sha256'] = sha256(a.out + '.npz')
    json.dump(info, open(a.out + '.json', 'w'), indent=1)
    print(a.set_name, n, counts, checks)

if __name__ == '__main__':
    main()
