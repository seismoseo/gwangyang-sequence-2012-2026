"""Per-channel template-vs-detection CC for every NEW and control detection (no day trace; ~0.5 s each) -> logs/similarity_channels.csv
(long format: origin, template_cuspid, cls, station, comp, phase, cc, weight). Basis for the waveform-support counts in nb39/nb37.
Classes: final catalogue (decision 20, 2026-10-06): control = a reference event detected by a template other than its own (not a
self-detection), new = not a reference event; as-run catalogues (2-15 Hz and mixed 5-20 Hz runs): control = in the DL catalog, not a
self-detection. Self-detections are skipped.
usage (scan env): python channel_cc.py"""
import os,sys,time
import numpy as np,pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import config as C
import similarity as S
ROOT=os.path.dirname(os.path.abspath(__file__));SUF='' if C.FMF_RUN in ('','full_period') else f'_{C.FMF_RUN}';OUT=os.path.join(ROOT,'logs',f'similarity_channels{SUF}.csv')
cat=pd.read_csv(C.CATALOG_CSV);known=cat.is_reference if 'is_reference' in cat else cat.in_DL
cat['cls']=np.where(cat.self_detection,'self',np.where(cat.is_new,'new',np.where(known,'control','in_template')))
sel=cat[cat.cls.isin(['new','control'])];sh=S.load_shelly();cusp=list(sh['cuspid'].astype(int));rows=[];t0=time.time()
for k,r in enumerate(sel.itertuples()):
    ti=cusp.index(int(r.template_cuspid))
    try:ch,net,mean=S.detection_vs_template(sh,ti,r.origin)
    except Exception as e:print('!!',r.origin,e,flush=True);continue
    for o in ch:rows.append(dict(origin=r.origin,template_cuspid=int(r.template_cuspid),cls=r.cls,catalog_cc=r.cc,station=o['station'],comp=o['comp'],phase=o['phase'],cc=o['cc'],weight=o.get('weight',np.nan)))
    if k%50==0:print(k,len(sel),f'{time.time()-t0:.0f} s',flush=True)
pd.DataFrame(rows).to_csv(OUT,index=False);print('wrote',len(rows),'channel rows ->',OUT)
