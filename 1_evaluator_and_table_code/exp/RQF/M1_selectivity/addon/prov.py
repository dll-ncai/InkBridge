"""Write / update PROVENANCE_<exp>.json (uhwr project rule).

python prov.py --exp ID --out PATH --run-dir DIR --code-tar TAR [--inputs f ...] [--checkpoints f ...]
               [--params JSON] [--seeds ...] [--command STR] [--venv DIR] [--outputs glob ...]
               [--notes STR] [--deviations STR ...] [--stage start|end]
Re-running with --stage end merges into the existing file: adds output sha256s, sacct states.
Times are UTC (Z).
"""
import argparse, glob, hashlib, json, os, subprocess, datetime, socket


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sacct(ids):
    out = []
    for j in ids:
        try:
            r = subprocess.run(['sacct', '-j', str(j), '-P', '-n', '--format=JobID,JobName,Partition,State,ExitCode,Start,End,Elapsed,AllocCPUS,MaxRSS,NodeList'],
                               capture_output=True, text=True, timeout=60).stdout
            out += [dict(zip('JobID JobName Partition State ExitCode Start_CEST End_CEST Elapsed AllocCPUS MaxRSS NodeList'.split(), l.split('|'))) for l in r.strip().splitlines()]
        except Exception as e:
            out.append({'JobID': str(j), 'error': str(e)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--run-dir', required=True); ap.add_argument('--code-tar')
    ap.add_argument('--inputs', nargs='*', default=[]); ap.add_argument('--checkpoints', nargs='*', default=[])
    ap.add_argument('--params', default='{}'); ap.add_argument('--seeds', nargs='*', default=[])
    ap.add_argument('--command', default=''); ap.add_argument('--venv', default='')
    ap.add_argument('--image', default=''); ap.add_argument('--outputs', nargs='*', default=[])
    ap.add_argument('--notes', default=''); ap.add_argument('--deviations', nargs='*', default=[])
    ap.add_argument('--slurm-ids', nargs='*', default=[]); ap.add_argument('--stage', default='start')
    a = ap.parse_args()
    P = json.load(open(a.out)) if os.path.exists(a.out) else {}
    P.setdefault('experiment', a.exp); P.setdefault('created_utc', now())
    P['run_dir'] = a.run_dir; P['updated_utc'] = now(); P['host_last'] = socket.gethostname()
    if a.code_tar:
        P['code_tarball'] = {'path': a.code_tar, 'sha256': sha(a.code_tar)}
    if a.inputs:
        P['inputs'] = [{'path': p, 'sha256': sha(p) if os.path.isfile(p) else None,
                        'resolved': os.path.realpath(p)} for p in a.inputs]
    if a.checkpoints:
        P['checkpoints'] = [{'path': p, 'resolved': os.path.realpath(p), 'sha256': sha(p)} for p in a.checkpoints]
    if a.params != '{}':
        P['parameters'] = json.loads(a.params)
    if a.seeds:
        P['seeds'] = a.seeds
    if a.command:
        P['command'] = a.command
    if a.image and os.path.exists(a.image):
        P['container'] = {'path': a.image, 'sha256': P.get('container', {}).get('sha256') or sha(a.image)}
    if a.venv:
        try:
            fr = subprocess.run(['apptainer', 'exec', '-B', '$SCRATCH_ROOT', a.image, os.path.join(a.venv, 'bin/python'), '-m', 'pip', 'freeze'],
                                capture_output=True, text=True, timeout=300).stdout if a.image else \
                 subprocess.run([os.path.join(a.venv, 'bin/python'), '-m', 'pip', 'freeze'], capture_output=True, text=True, timeout=300).stdout
            P['environment_freeze'] = {'venv': a.venv, 'pip_freeze': fr.strip().splitlines()}
        except Exception as e:
            P['environment_freeze'] = {'venv': a.venv, 'error': str(e)}
    ids = sorted(set(P.get('slurm_job_ids', []) + a.slurm_ids + ([os.environ['SLURM_JOB_ID']] if os.environ.get('SLURM_JOB_ID') else [])))
    P['slurm_job_ids'] = ids
    P['slurm_states_' + a.stage] = {'recorded_utc': now(), 'note': 'sacct times are CEST (UTC+2)', 'rows': sacct(ids)}
    if a.outputs:
        files = sorted({f for g in a.outputs for f in glob.glob(g, recursive=True) if os.path.isfile(f)})
        P['outputs'] = [{'path': f, 'bytes': os.path.getsize(f), 'sha256': sha(f)} for f in files]
    if a.notes:
        P.setdefault('notes', []).append(a.notes) if isinstance(P.get('notes'), list) else P.update(notes=[a.notes])
    if a.deviations:
        P['deviations'] = a.deviations
    json.dump(P, open(a.out, 'w'), indent=1)
    print('provenance', a.out, 'outputs', len(P.get('outputs', [])))


if __name__ == '__main__':
    main()
