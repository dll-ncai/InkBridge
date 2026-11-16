#!/usr/bin/env python3
"""RQF R3 summary: per (base,set) ORDER-WIN (sealed_read.py A1 rule, unchanged), random-of-3 control check (rnd3-pool3: every metric CI contains 0),
plus family Holm on top: p_cell = min(1, 3*min p_boot over {penup_f1_v2,strict_order_acc,direction_acc}) (bootstrap B=2000; floor 1/2000 -> p reported as max(p,1/2000)).
R3a family = 5 bases on test_ouhdl (Holm over 5); R3b family = 6 sets within each base (Holm within base). Usage: rq3_summarize.py READ_DIR OUT_PREFIX"""
import json,glob,os,sys,pandas as pd
d,o=sys.argv[1:3]; R=[]
for f in sorted(glob.glob(d+'/cells_*.json')):
    lab=os.path.basename(f)[6:-5]; base,s=lab.split('__',1); J=json.load(open(f)); P=pd.read_csv(f'{d}/paired_{lab}.csv')
    c=J['cells']['COMP_rep3__vs__COMP_pool3']; m=c['metrics']; k=J['cells']['COMP_rnd3__vs__COMP_pool3']['metrics']
    ctrl=all(v['ci_lo']<=0<=v['ci_hi'] for v in k.values())
    pm=min(max(m[x]['p_boot'],1/2000) for x in ('penup_f1_v2','strict_order_acc','direction_acc'))
    R.append(dict(base=base,set=s,order_win=c['order_win'],part1=','.join(c['part1_metrics']),control_all_ci_contain0=ctrl,n_units=m['aiou']['n'],p_cell=min(1.0,3*pm),
      **{f'd_{x}':m[x]['delta'] for x in ('penup_f1_v2','strict_order_acc','direction_acc','aiou')},**{f'lo_{x}':m[x]['ci_lo'] for x in ('penup_f1_v2','strict_order_acc','direction_acc','aiou')},
      **{f'hi_{x}':m[x]['ci_hi'] for x in ('penup_f1_v2','strict_order_acc','direction_acc','aiou')}))
D=pd.DataFrame(R)
def holm(p):
    o_=sorted(range(len(p)),key=lambda i:p[i]); adj=[0]*len(p); run=0
    for r,i in enumerate(o_): run=max(run,min(1.0,(len(p)-r)*p[i])); adj[i]=run
    return adj
D['p_holm']=float('nan'); D['family']=''
a=D[(D.set=='test_ouhdl')&D.base.isin(['E0','M5','LUOft_p10','NOFM_W_FT','PEINIT_W_FT'])].index
if len(a): D.loc[a,'p_holm']=holm(D.loc[a,'p_cell'].tolist()); D.loc[a,'family']='R3a(5 bases)'
for b in D.base.unique():
    ix=D[(D.base==b)&(D.set!='test_ouhdl')].index
    if len(ix): D.loc[ix,'p_holm']=holm(D.loc[ix,'p_cell'].tolist()); D.loc[ix,'family']='R3b('+b+')'
D['order_win_and_holm']=D.order_win&(D.p_holm<0.05)&D.control_all_ci_contain0
D.to_csv(o+'_cells.csv',index=False)
print(D.groupby('family').agg(cells=('set','size'),order_win=('order_win','sum'),win_holm_ctrl=('order_win_and_holm','sum')).to_string())
