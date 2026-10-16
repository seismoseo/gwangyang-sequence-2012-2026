"""Compare the 5-20 Hz FMF run with the 2-15 Hz run over the days both have scanned: detections matched within +-4 s,
per-class counts, CC change of common detections, detections unique to each run, and the fate of the named candidates.
usage: python compare_runs.py [RUN_B=full_period_5_20H] -> prints a summary, writes logs/compare_<RUN_B>.csv
Classes: final catalogue (full_period_5_20H, decision 20, 2026-10-06) self / reference / new; as-run catalogues (2-15 Hz and mixed
5-20 Hz runs) self / in_DL / in_template / new. The two class sets are printed side by side, not mapped onto each other."""
import os,sys,glob
import numpy as np,pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__));A='full_period';B=sys.argv[1] if len(sys.argv)>1 else 'full_period_5_20H';TOL=4.0   # default: the final run (the 5-20 Hz pilot full_period_5_20 never had a catalog built)
def classes(c):
    k=np.where(c.is_reference,'reference','in_template') if 'is_reference' in c else np.where(c.in_DL,'in_DL','in_template')
    return np.where(c.self_detection,'self',np.where(c.is_new,'new',k))
def load(run):
    c=pd.read_csv(f'{HERE}/catalogs/fmf_haenam_catalog_{run}_seg.csv',parse_dates=['origin']);c['cls']=classes(c);c['day']=c.origin.dt.strftime('%Y-%j');return c
days=lambda run:{os.path.basename(f)[4:12] for f in glob.glob(f'{HERE}/catalogs/daily_shelly_{run}_seg/det_*.parquet')}
common=days(A)&days(B);print(f'days scanned by both runs: {len(common)} (B has {len(days(B))})')
a=load(A);b=load(B);a=a[a.day.str.replace("-","_").isin(common)];b=b[b.day.str.replace("-","_").isin(common)]
print('\ncounts over the common days:');print(pd.DataFrame({A:a.cls.value_counts(),B:b.cls.value_counts()}).fillna(0).astype(int).to_string())
ta=a.origin.values.astype('datetime64[ns]').astype('int64')/1e9;tb=b.origin.values.astype('datetime64[ns]').astype('int64')/1e9
ia=np.argsort(ta);tas=ta[ia];match=np.full(len(b),-1)
for k,t in enumerate(tb):
    j=np.searchsorted(tas,t);cands=[jj for jj in (j-1,j) if 0<=jj<len(tas) and abs(tas[jj]-t)<=TOL]
    if cands:match[k]=ia[min(cands,key=lambda jj:abs(tas[jj]-t))]
b['match_A']=match;m=b[b.match_A>=0].copy();m['cc_A']=a.cc.values[m.match_A.values];m['cls_A']=a.cls.values[m.match_A.values];m['dcc']=m.cc-m.cc_A
print(f'\nmatched within {TOL:.0f} s: {len(m)} of {len(b)} B detections; A detections without a B match: {len(a)-m.match_A.nunique()}')
print('CC change (B - A) of matched detections by class in A:');print(m.groupby('cls_A').dcc.describe()[['count','mean','50%','25%','75%']].round(3).to_string())
print('\nB-only detections by class:',b[b.match_A<0].cls.value_counts().to_dict());print('A-only detections by class:',a[~a.index.isin(m.match_A)].cls.value_counts().to_dict())
for o in ['2023-06-17 19:30:26','2023-01-12 07:43:21','2020-10-19 18:53:33','2020-10-16 20:29:11','2024-04-21 02:07:32','2026-04-29 18:47:09']:
    t=pd.Timestamp(o);hb=b[(b.origin-t).abs()<pd.Timedelta(seconds=TOL)];ha=a[(a.origin-t).abs()<pd.Timedelta(seconds=TOL)]
    if len(ha) or len(hb):print(f'  {o}: A cc {ha.cc.max() if len(ha) else float("nan"):.2f} ({ha.n_present.max() if len(ha) else 0} sta, {ha.n_templates.max() if len(ha) else 0} tpl) | B cc {hb.cc.max() if len(hb) else float("nan"):.2f} ({hb.n_present.max() if len(hb) else 0} sta, {hb.n_templates.max() if len(hb) else 0} tpl)')
b.to_csv(f'{HERE}/logs/compare_{B}.csv',index=False);print('->',f'{HERE}/logs/compare_{B}.csv')
