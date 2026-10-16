"""Compare two FMF runs on chosen days from their built catalogs (run_*.sh catalog): per-class counts, detections matched within
+-4 s with their CC change, and detections unique to each run. usage: python compare_days.py RUN_A RUN_B 2020-124 2026-228 ...
(RUN names as in catalogs/fmf_haenam_catalog_<run>_seg.csv) -> prints tables, writes logs/compare_days_<RUN_B>.csv
Classes: final catalogue (full_period_5_20H, decision 20, 2026-10-06) self / reference / new; as-run catalogues (2-15 Hz and mixed
5-20 Hz runs) self / in_DL / in_template / new. The two class sets are printed side by side, not mapped onto each other."""
import os,sys
import numpy as np,pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__));A,B=sys.argv[1],sys.argv[2];DAYS=sys.argv[3:];TOL=4.0
def classes(c):
    k=np.where(c.is_reference,'reference','in_template') if 'is_reference' in c else np.where(c.in_DL,'in_DL','in_template')
    return np.where(c.self_detection,'self',np.where(c.is_new,'new',k))
def load(run):
    c=pd.read_csv(f'{HERE}/catalogs/fmf_haenam_catalog_{run}_seg.csv',parse_dates=['origin']);c['cls']=classes(c);c['day']=c.origin.dt.strftime('%Y-%j');return c[c.day.isin(DAYS)].reset_index(drop=True)
a,b=load(A),load(B);pd.set_option('display.width',250)
print(f'days {DAYS}: {A} {len(a)} declustered detections, {B} {len(b)}');print(pd.DataFrame({A:a.cls.value_counts(),B:b.cls.value_counts()}).fillna(0).astype(int).to_string())
ta=a.origin.values.astype('datetime64[ns]').astype('int64')/1e9;tb=b.origin.values.astype('datetime64[ns]').astype('int64')/1e9;ia=np.argsort(ta);tas=ta[ia];match=np.full(len(b),-1)
for k,t in enumerate(tb):
    j=np.searchsorted(tas,t);cands=[jj for jj in (j-1,j) if 0<=jj<len(tas) and abs(tas[jj]-t)<=TOL]
    if cands:match[k]=ia[min(cands,key=lambda jj:abs(tas[jj]-t))]
b['match_A']=match;m=b[b.match_A>=0].copy();m['cc_A']=a.cc.values[m.match_A.values];m['cls_A']=a.cls.values[m.match_A.values];m['n_present_A']=a.n_present.values[m.match_A.values];m['dcc']=m.cc-m.cc_A
print(f'\nmatched within {TOL:.0f} s: {len(m)} of {len(b)} {B} detections; {A}-only: {len(a)-m.match_A.nunique()}')
print(f'CC change ({B} - {A}) of matched detections by class in {A}:');print(m.groupby('cls_A').dcc.describe()[['count','mean','25%','50%','75%']].round(3).to_string())
print(f'\n{B}-only detections by class:',b[b.match_A<0].cls.value_counts().to_dict(),'| CC median',round(float(b[b.match_A<0].cc.median()),3) if (b.match_A<0).any() else None)
print(f'{A}-only detections by class:',a[~a.index.isin(m.match_A)].cls.value_counts().to_dict(),'| CC median',round(float(a[~a.index.isin(m.match_A)].cc.median()),3) if len(a)-m.match_A.nunique() else None)
print(f'\ntop {B}-only detections:');print(b[b.match_A<0].sort_values('cc',ascending=False)[['origin','template_cuspid','cc','n_present','n_templates','cls']].head(8).to_string(index=False))
b.to_csv(f'{HERE}/logs/compare_days_{B}.csv',index=False)
