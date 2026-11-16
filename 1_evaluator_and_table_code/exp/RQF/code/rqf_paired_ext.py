
"""RQF R3-task2: paired writer-cluster CIs, external baselines vs nominees (OUHD-L test, native). Code reuses d3a_tables.py bootstrap (default_rng(0) multinomial, ratio of sums, B=2000)."""
import glob, hashlib, json, os, sys, re
import numpy as np, pandas as pd
E="$U/exp/"; OUT=sys.argv[1]; B=2000
os.makedirs(OUT,exist_ok=True)
MET=["aiou","penup_f1_v2","strict_order_acc","direction_acc","kendall_tau","tau_cp","ldtw"]
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<22),b""): h.update(b)
    return h.hexdigest()
def boot_counts(n,B,seed=0):
    return np.random.default_rng(seed).multinomial(n,np.full(n,1.0/n),size=B).astype(float)
def boot(v,wix,nw,C):
    ok=~np.isnan(v); m=float(v[ok].mean()) if ok.any() else np.nan
    if not ok.any() or C is None: return m,np.nan,np.nan,int(ok.sum())
    S=np.bincount(wix[ok],weights=v[ok],minlength=nw); N=np.bincount(wix[ok],minlength=nw).astype(float)
    num,den=C@S,C@N; bm=num[den>0]/den[den>0]; lo,hi=np.percentile(bm,[2.5,97.5]); return m,float(lo),float(hi),int(ok.sum())
def load(files, flt=None):
    dfs=[]
    for f in files:
        d=pd.read_csv(f,low_memory=False); d["_file"]=os.path.basename(f)
        if "seed" not in d.columns: d["seed"]=0
        dfs.append(d)
    d=pd.concat(dfs,ignore_index=True)
    if flt is not None: d=d[flt(d)]
    return d
def peritem(d,key="unit_id"):
    d=d[d["protocol"]=="native"].copy()
    d[key]=d[key].astype(str)
    cols=[c for c in ["aiou","penup_f1_v2","strict_order_acc","direction_acc","kendall_tau","ldtw","n_gt_strokes"] if c in d.columns]
    g=d.groupby(key,sort=True)
    a=g[cols].mean(numeric_only=True); a["writer"]=g["writer_id"].first().astype(str)
    a["tau_cp"]=np.where(a["n_gt_strokes"]>=2,np.nan_to_num(a["kendall_tau"].to_numpy(float),nan=0.0),np.nan)
    a["n_rows"]=g.size(); return a
prov={"script":"rqf_paired_ext.py","B":B,"seed":0,"methods":{}}
M={}
def reg(name,files,flt=None,device="",note="",key="unit_id"):
    d=load(files,flt); a=peritem(d,key); M[name]=a
    prov["methods"][name]=dict(device=device,note=note,n_files=len(files),n_units=len(a),n_writers=int(a.writer.nunique()),
      files_sha256={os.path.basename(f):sha(f) for f in files[:40]},rows_per_unit_median=float(a.n_rows.median()))
    print(name,len(files),len(a),flush=True)
g=lambda p:sorted(glob.glob(E+p,recursive=True))
# nominees / comparators (test_ouhdl rows031)
reg("lineHT_d4",g("CAND/method_round/ConfirmV2/score_out/4090/**/test_ouhdl__lineHT_d4*__rows031.csv"),device="RTX 4090",note="3 train seeds x 3 sampling seeds, per-unit mean",)
reg("LineT_d4",g("CAND/method_round/ConfirmV2/score_out/4090/**/test_ouhdl__lineT_d4*__rows031.csv"),device="RTX 4090",note="3 train seeds x 3 sampling seeds")
reg("COMP_rep3",g("ORDER/sealed/COMP_rep3/score_out/**/test_ouhdl__*_rep3__seed0__rows031.csv"),device="RTX 4090 (verifier on CPU)",note="verifier output, 3 training seeds, seed0 file each")
reg("NOFM_W_FT",g("FMD/xs/runs/fmdx_monitor/score_out/**/test_ouhdl__NOFM_W_ft_ms1__seed*__rows031.csv"),device="RTX A4000",note="1 FT seed, 3 sampling seeds")
reg("PEINIT_W_FT",g("FMD/xs/runs/fmdx_monitor/score_out/**/test_ouhdl__PEINIT_W_ft_ms2__seed*__rows031.csv"),device="RTX A4000",note="1 FT seed, 3 sampling seeds")
# external baselines
reg("E0scratch_p01",g("D2_MOE/controls/score_out/rows/test_ouhdl__E0scratch_p01_ms10__seed*__rows031.csv"),device="CLUSTER-A GPU",note="E0 from scratch on p01, ms10, 3 sampling seeds")
reg("PEN-Net",[E+"PENNET/results/test_rendered_ms10_rows.csv"],device="n/a",note="greedy ms10 rendered input; evaluator per PENNET/results/uhwr_eval_version.txt")
reg("InkSight_e0r",[E+"RB4_baselines/inksight_e0r/rows/inksight_e0r_rendered_rows.csv"],note="Small-p zero-shot, e0r crops, rendered")
reg("InkSight_g1f4",[E+"RB4_baselines/inksight_e0r/rows/inksight_g1f4_rendered_rows.csv"],note="Small-p zero-shot, g1 f=4 crops, rendered")
NOM=["lineHT_d4","LineT_d4","COMP_rep3","NOFM_W_FT","PEINIT_W_FT"]; BASE=["InkSight_e0r","InkSight_g1f4","PEN-Net","E0scratch_p01"]
common=set.intersection(*[set(M[k].index) for k in NOM+BASE]); print("common",len(common),flush=True)
rows=[]; absr=[]
def run(basis,units_fn,pairs):
    for b,n in pairs:
        u=units_fn(b,n); A=M[n].loc[u]; Bm=M[b].loc[u]
        uw,wix=np.unique(A["writer"].to_numpy(),return_inverse=True); C=boot_counts(len(uw),B) if len(uw)>1 else None
        for met in MET:
            if met not in A or met not in Bm: continue
            va,vb=A[met].to_numpy(float),Bm[met].to_numpy(float); d=va-vb
            ma=boot(va,wix,len(uw),C); mb=boot(vb,wix,len(uw),C); md=boot(d,wix,len(uw),C)
            rows.append(dict(basis=basis,nominee=n,baseline=b,metric=met,n_units=len(u),n_writers=len(uw),n_defined=md[3],nominee_mean=ma[0],baseline_mean=mb[0],delta=md[0],ci_lo=md[1],ci_hi=md[2]))
pairs=[(b,n) for b in BASE for n in NOM]
run("common_all_methods",lambda b,n:sorted(common),pairs)
run("pairwise_intersection",lambda b,n:sorted(set(M[b].index)&set(M[n].index)),pairs)
T=pd.DataFrame(rows); T.to_csv(OUT+"/paired_ext_baselines_ouhdl_unit.csv",index=False)
# absolute on common
for k in NOM+BASE:
    A=M[k].loc[sorted(common)]; uw,wix=np.unique(A["writer"].to_numpy(),return_inverse=True); C=boot_counts(len(uw),B)
    r=dict(method=k,n_units=len(A),n_writers=len(uw))
    for met in MET:
        m=boot(A[met].to_numpy(float),wix,len(uw),C); r[met]=m[0]; r[met+"_lo"]=m[1]; r[met+"_hi"]=m[2]; r[met+"_ndef"]=m[3]
    absr.append(r)
pd.DataFrame(absr).to_csv(OUT+"/absolute_common_units.csv",index=False)
# TRACE: line level
tf=E+"L4_TRACE/out/per_line_L4_TRACE_uhwr_eval031.csv"; t=pd.read_csv(tf,low_memory=False)
print("TRACE datasets",t["dataset"].unique(),"arms",t["input_arm"].unique(),"scripts",t["script"].unique(),flush=True)
t=t[t["dataset"].astype(str).str.contains("ouhd",case=False)]
trace_rows=[]
lines={}
for n in NOM:  # nominee units -> line means
    d=M[n].copy(); d["line"]=d.index.str.replace(r"_[LG]\d+$","",regex=True)
    lines[n]=d
for arm in sorted(t["input_arm"].unique()):
    ta=t[(t["input_arm"]==arm)&(t["protocol"]=="native")].copy(); ta["sample_id"]=ta["sample_id"].astype(str)
    if "penup_f1_v2" not in ta: continue
    ta=ta.groupby("sample_id").agg({**{m:"mean" for m in ["aiou","penup_f1_v2","kendall_tau","direction_acc","ldtw"] if m in ta},"n_gt_strokes":"first","writer_id":"first"})
    ta["writer"]=ta["writer_id"].astype(str)
    for n in NOM:
        d=lines[n]; g=d.groupby("line")[[m for m in ["aiou","penup_f1_v2","kendall_tau","direction_acc","ldtw"]]].mean(numeric_only=True)
        u=sorted(set(ta.index)&set(g.index))
        if not u: print("no overlap",arm,n); continue
        uw,wix=np.unique(ta.loc[u,"writer"].to_numpy(),return_inverse=True); C=boot_counts(len(uw),B)
        for met in ["aiou","penup_f1_v2","kendall_tau","direction_acc","ldtw"]:
            if met not in ta: continue
            va,vb=g.loc[u,met].to_numpy(float),ta.loc[u,met].to_numpy(float)
            md=boot(va-vb,wix,len(uw),C); trace_rows.append(dict(basis="line_level_mean_of_units",nominee=n,baseline="TRACE_"+arm,metric=met,n_lines=len(u),n_writers=len(uw),n_defined=md[3],nominee_mean=float(np.nanmean(va)),baseline_mean=float(np.nanmean(vb)),delta=md[0],ci_lo=md[1],ci_hi=md[2]))
pd.DataFrame(trace_rows).to_csv(OUT+"/paired_trace_line_level.csv",index=False)
prov["common_units"]=len(common); prov["trace_file"]=dict(path=tf,sha256=sha(tf))
prov["note_trace"]="TRACE is line-level (line protocol, ink height 56 px); nominee unit rows averaged per line; not same-protocol, descriptive"
json.dump(prov,open(OUT+"/PROVENANCE_paired_ext.json","w"),indent=1)
print("done",flush=True)
