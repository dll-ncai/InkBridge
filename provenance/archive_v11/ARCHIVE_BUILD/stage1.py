#!/usr/bin/env python3
import os,glob,hashlib,shutil,json,time,subprocess,sys
U='$U'; S=U+'/tmp/rv11_archive/stage1'
os.makedirs(S+'/src',exist_ok=True); os.chdir(U)
CAP=3*1024*1024
BAD=('.npz','.pt','.pkl','.err','.lock','.pyc','.b64','.png','.pdf','.sif')
ALLOW_BIG=['exp/RQF/M1_selectivity/out/e1_v2_scores_per_unit.csv.gz']
G="""shared/uhwr_eval-0.3.1/**
exp/D3_ROUTE/routeA/code/*
exp/D3_ROUTE/routeA/results/*
exp/D3_ROUTE/routeA/routeA_preds_index.csv
exp/D3_ROUTE/routeA/SHA256SUMS_routeA_preds.txt
exp/D3_ROUTE/routeA/jobids.txt
exp/D3_ROUTE/stageA/results/*
exp/D3_ROUTE/router/*
exp/D3_ROUTE/SHA256SUMS.txt
exp/RQF/code/rqf_v1/*
exp/RQF/code/sm_v1/*
exp/RQF/code/reviewer_v1/*.py
exp/RQF/code/rq3_v1/*
exp/RQF/code/rq3_split_v1/*
exp/RQF/code/rqf1_v1/*
exp/RQF/code/rqf_paired_ext.py
exp/RQF/code/rqf_pairext.sbatch
exp/RQF/code/sm_v1.tgz
exp/RQF/admin/*
exp/RQF/plans/*
exp/RQF/queue/*
exp/RQF/tables/**
exp/RQF/PROVENANCE_RQF_R2.json
exp/RQF/R3/*.json
exp/RQF/R3/*.csv
exp/RQF/R3/*.py
exp/RQF/R3/read_all3/*
exp/RQF/R3/read_A/*
exp/RQF/R3/read_C/*
exp/RQF/R3/read_C3/*
exp/RQF/R3/split_v1/*
exp/RQF/R3/rep/*
exp/RQF/R1/PROVENANCE_R1.json
exp/RQF/R1d/plan.json
exp/RQF/R1d/code/*
exp/RQF/R1d/read/*
exp/RQF/R1d/read/*/*
exp/RQF/M3_descriptive/*
exp/RQF/M1_selectivity/code/**
exp/RQF/M1_selectivity/addon/*
exp/RQF/M1_selectivity/out/*
exp/RQF/M1_selectivity/bundle/*.json
exp/RQF/score_out/job_history.txt
exp/RQF/score_out/score_env_*
exp/RQF/score_out/score_summary_*
exp/RQF/inventory/*.tsv
exp/RQF/inventory/*.txt
exp/RQF/runs/*/run_config.json
exp/RQF/runs/*/job_history.txt
exp/RQF/runs/*/training_complete.json
exp/RQF/runs/*/*/run_config.json
exp/RQF/runs/*/*/job_history.txt
exp/RQF/runs/*/*/training_complete.json
exp/RQF/R2d/admin/*
exp/RQF/R2d/code/*
exp/RQF/R2d/code/*/*
exp/RQF/R2d/runs/*/run_config.json
exp/RQF/R2d/runs/*/training_complete.json
exp/RQF/R2d/runs/*/*/run_config.json
exp/FMD/*.json
exp/FMD/*.md
exp/FMD/plans/*
exp/FMD/gate/*
exp/FMD/screen/*
exp/FMD/runs/*.md
exp/FMD/runs/*.sha256
exp/FMD/xs/*.json
exp/FMD/xs/*.md
exp/FMD/xs/*.txt
exp/FMD/xs/plans/*
exp/FMD/xs/admin/*
exp/FMD/xs/read_v1/*
exp/FMD/xs/read_v1/code/*
exp/FMD/xs/screen/*
exp/FMD/xs/runs/*.md
exp/FMD/xs/runs/*.sha256
exp/FMD/xs/runs/*/run_config.json
exp/FMD/xs/runs/*/job_history.txt
exp/FMD/xs/runs/*/training_complete.json
exp/FMD/xs/runs/fmdx_monitor/*.md
exp/FMD/xs/runs/fmdx_monitor/*.json
exp/FMD/xs/runs/fmdx_monitor/*.csv
exp/FMD/xs/runs/fmdx_monitor/*.tsv
exp/FMD/xs/runs/fmdx_monitor/tables*/*
exp/FMD/xs/code/fmdx_monitor_v1/*
exp/FMD/xs/code/fmdx_backbone_v1/*
exp/FMD/xs/code/fmdx_backbone_v1/jobs/*
exp/FMD/xs/code/fmdx_backbone_v1/controller/*
exp/M5/PROVENANCE_M5.json
exp/M5/code_tarballs/*
exp/M5/jobs/*
exp/M5/report/*
exp/M5/runs/*/run_config.json
exp/M5/runs/*/job_history.txt
exp/M5/runs/*/training_complete.json
exp/M5/runs/*/*/run_config.json
exp/M5/runs/*/*/job_history.txt
exp/M5/runs/*/*/training_complete.json
exp/PENNET/*.json
exp/PENNET/*.txt
exp/PENNET/*.tar.gz
exp/PENNET/results/*
exp/D2_MOE/code/*.tar.gz
exp/D2_MOE/results/*
exp/CAND/method_round/COMP/*
exp/CAND/method_round/COMP/runs/*/run_config.json
exp/CAND/method_round/COMP/runs/*/training_complete.json
exp/CAND/method_round/*.json
exp/CAND/method_round/*.md
exp/X4a/*.json
exp/X4a/data/*.md
exp/X4a/data/*.json
data/m5_multiscript_v1/*
data/d2_v2/*/*
data/ouhdl_ligatures_v3/PROVENANCE.json
data/ouhdl_ligatures_v3/README.md
data/gold_lines_v0_sample_ids.txt
""".split()
src=open(S+'/SOURCES.tsv','w'); skp=open(S+'/SKIPPED.tsv','w'); n=0; tot=0; seen=set()
for g in G:
    fs=glob.glob(g,recursive=True)
    if not fs: skp.write('NOMATCH\t%s\t0\n'%g)
    for f in sorted(fs):
        if f in seen or not os.path.isfile(f) or os.path.islink(f) and not os.path.exists(f): continue
        seen.add(f); sz=os.path.getsize(f); b=os.path.basename(f)
        if '__pycache__' in f or b.startswith('slurm-') or '.stale_' in f: continue
        if f.endswith(BAD): skp.write('TYPE\t%s\t%d\n'%(f,sz)); continue
        if sz>CAP and f not in ALLOW_BIG: skp.write('SIZE\t%s\t%d\n'%(f,sz)); continue
        h=hashlib.sha256(open(f,'rb').read()).hexdigest()
        d=S+'/src/'+f; os.makedirs(os.path.dirname(d),exist_ok=True); shutil.copyfile(f,d)
        src.write('%s\t%s\t%d\t%s\n'%(f,h,sz,time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(os.path.getmtime(f))))); n+=1; tot+=sz
src.close(); skp.close()
# rows listings + headers
with open(S+'/ROWS_LISTING.txt','w') as o:
    for d in ['exp/RQF/score_out/rows','exp/FMD/xs/runs/fmdx_monitor/score_out/rows','exp/FMD/xs/runs/fmdx_monitor/score_out_cmp/rows','exp/CAND/method_round/COMP/score_out/4090/rows','exp/CAND/method_round/COMP/score_out','exp/D3_ROUTE/routeA/score_out/rows','exp/D3_ROUTE/stageA/score_out/rows','exp/RQF/R3/score_out','exp/RQF/R3/score_out/rows','exp/RQF/R3/comp_a4000','exp/RQF/R1d/score_out','exp/RQF/R1d/score_out/rows','exp/RQF/R1/score_out','exp/RQF/R1/score_out/rows','exp/RQF/R1d/read','exp/RQF/R3/rep','exp/RQF/R3/read_all3','exp/FMD/xs/runs/fmdx_monitor','exp/CAND/method_round/COMP','exp/CAND/method_round']:
        o.write('## %s\n'%d)
        try:
            ents=sorted(os.listdir(d))
        except Exception as e:
            o.write('ERR %s\n'%e); continue
        o.write('n=%d\n'%len(ents))
        for e in ents:
            p=d+'/'+e
            try: o.write('%d\t%s%s\n'%(os.path.getsize(p),e,'/' if os.path.isdir(p) else ''))
            except Exception: pass
        cs=[e for e in ents if e.endswith('.csv')]
        if cs:
            with open(d+'/'+cs[0],errors='replace') as fh: o.write('HEAD %s\n%s'%(cs[0],''.join([fh.readline() for _ in range(3)])))
# npz id dumps
rep={}
try:
    import numpy as np
    NP=['data/m5_multiscript_v1/m5_ft_ouhdl_p01.npz','data/m5_multiscript_v1/m5_ft_ouhdl_p10.npz','data/m5_multiscript_v1/m5_ft_ouhdl_p30.npz','data/m5_multiscript_v1/m5_test_iam500.npz']+sorted(glob.glob('data/m5_multiscript_v1/m5_val_*.npz'))+sorted(glob.glob('exp/X4a/data/x4a_*.npz'))+['data/d2_v2/unipen_de/e0_test.npz','data/d2_v2/onhw/e0_all.npz','data/ouhdl_e0_v3/e0_test.npz','data/ouhdl_e0_v3/e0_val.npz','data/ouhdl_e0_v3/e0_gold.npz','data/ouhdl_e0_v3/e0_train.npz']+sorted(glob.glob('exp/RQF/data/render224/*.json'))
    os.makedirs(S+'/ids',exist_ok=True)
    for f in NP:
        if f.endswith('.json'):
            shutil.copyfile(f,S+'/ids/'+os.path.basename(f)); continue
        try:
            z=np.load(f,allow_pickle=True); info={}; cols={}
            import zipfile
            zi={i.filename[:-4]:i.file_size for i in zipfile.ZipFile(f).infolist()}
            for k in z.files:
                if zi.get(k,0)>40*1024*1024: info[k]='big %d'%zi[k]; continue
                a=z[k]; info[k]=[str(a.dtype),list(a.shape)]
                if a.ndim==1 and a.dtype.kind in 'USOiu' and a.shape[0]>1 and not any(t in k.lower() for t in ('offset','len','seq','ink','img','image','render','point')):
                    cols[k]=a
                if a.ndim==0: info[k].append(str(a)[:300])
            rep[f]={'sha256':hashlib.sha256(open(f,'rb').read()).hexdigest() if os.path.getsize(f)<200e6 else 'not hashed (size)','size':os.path.getsize(f),'mtime':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(os.path.getmtime(f))),'keys':info,'dumped':sorted(cols)}
            ns=set(len(v) for v in cols.values())
            for nn in ns:
                ks=[k for k in sorted(cols) if len(cols[k])==nn]
                with open(S+'/ids/'+f.replace('/','__')+'.n%d.ids.tsv'%nn,'w') as o:
                    o.write('\t'.join(ks)+'\n')
                    for i in range(nn): o.write('\t'.join(str(cols[k][i]).replace('\t',' ').replace('\n',' ') for k in ks)+'\n')
        except Exception as e: rep[f]={'error':repr(e)}
except Exception as e: rep['numpy']=repr(e)
json.dump(rep,open(S+'/NPZ_KEYS.json','w'),indent=1)
subprocess.call('cd %s/.. && tar czf stage1.tar.gz stage1 && sha256sum stage1.tar.gz > stage1.tar.gz.sha256 && ls -l stage1.tar.gz >> stage1.tar.gz.sha256'%S,shell=True)
print('DONE files',n,'bytes',tot)
