"""Quiescence bursts (nb45 input): repeater gathers, key-detection figures and relative magnitudes for the detections around three
quiescence-era reference events, from the refined 5-20 Hz run. usage (run env of full_period_5_20H): python burst_figs.py
Relative size of each detection against the burst template: per S channel of the template, the least-squares amplitude ratio of the
detection's S window on the template's S window at the lag (|lag| <= 0.3 s) of best correlation; channels kept when that correlation
is >= CC_MIN and both windows have SNR >= SNR_MIN over pre-event noise (origin-6 to origin-1 s, same channel); dM = median log10 r
(slope 1, as nb43). Absolute ML: from any detection in the burst that is a reference event with an nb43 magnitude (anchor).
Classes are the final catalog's (decision 20): a detection is a reference event (is_reference, reference_event_id) or new; every
catalog detection in a burst window is kept (no extra de-duplication)."""
import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # the project registry, found upward from this file
import os,sys,shutil,subprocess

def _link(src, dst):   # decision 25 (one home): dst becomes a relative link to the product src, never a byte copy
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.relpath(src, os.path.dirname(os.path.abspath(dst))), dst)
import numpy as np,pandas as pd
from obspy import UTCDateTime
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import config as C,similarity as S
from station_gather import station_gather
HERE=os.path.dirname(os.path.abspath(__file__));ROOT=os.path.dirname(HERE)
OUT=str(haenam.ROOT / 'analysis/seismicity/figs_pub/joint_quiescence_bursts');os.makedirs(OUT,exist_ok=True)
CC_MIN=0.6;SNR_MIN=2.0;LAG=30
BURSTS=[dict(key='B2020',name='2020-10-17 burst',tpl=400171,station='KS.JDO2',t0='2020-10-15',t1='2020-10-29'),
        dict(key='B2026a',name='2026-01-30 burst',tpl=400180,station='JU.JN01',t0='2026-01-30 20:00',t1='2026-01-31 02:00'),
        dict(key='B2026b',name='2026-04-29 burst',tpl=400181,station='JU.JN01',t0='2026-04-29 12:00',t1='2026-05-04')]
sh=S.load_shelly();cusp=list(sh['cuspid'].astype(int));stations=[str(x) for x in sh['stations']];prefixes=[str(p) for p in sh['channels']];comps=[str(c) for c in sh['components']]
cat=pd.read_csv(C.CATALOG_CSV,parse_dates=['origin'],dtype={'reference_event_id':str});cat['reference_event_id']=cat.reference_event_id.fillna('').str.replace(r'\.0$','',regex=True)
mag=pd.read_csv(haenam.path(None, 'ml_s_table'),dtype={'event_id':str},parse_dates=['time'])

def rms(x):return float(np.sqrt(np.mean(np.square(x-x.mean())))) if x is not None and x.size else np.nan
def relsize(ti,origin):
    o=UTCDateTime(pd.Timestamp(origin).to_pydatetime());to=UTCDateTime(str(sh['origin_utc'][ti]));rows=[]
    for si,ph,ci,mo,n,tw,j in S.channels_of(sh,ti):
        if ph!='S':continue
        net,sta=stations[si].split('.');st=o+float(sh['ref_s'][ti])+mo/C.FS
        d=S.read_window(net,sta,comps[ci],prefixes[si],st-LAG/C.FS,n+2*LAG)
        nz=S.read_window(net,sta,comps[ci],prefixes[si],o-6.0,int(5*C.FS));nt=S.read_window(net,sta,comps[ci],prefixes[si],to-6.0,int(5*C.FS))
        if d is None or nz is None or nt is None:continue
        t=tw-tw.mean();best=(0.0,LAG,np.nan)
        for L in range(2*LAG+1):
            x=d[L:L+n];x=x-x.mean();den=np.sqrt((x*x).sum()*(t*t).sum())
            if den>0:
                cc=float((x*t).sum()/den)
                if abs(cc)>abs(best[0]):best=(cc,L,float((x*t).sum()/(t*t).sum()))
        cc,L,r=best;x=d[best[1]:best[1]+n]
        rows.append(dict(station=stations[si],comp=comps[ci],cc=cc,lag_s=(L-LAG)/C.FS,ratio=r,snr_det=rms(x)/rms(nz),snr_tpl=rms(tw)/rms(nt)))
    return pd.DataFrame(rows)

allrows=[];chanrows=[];SETS=[];GROWS=[]
for B in BURSTS:
    ti=cusp.index(B['tpl']);det=cat[(cat.origin>=B['t0'])&(cat.origin<=B['t1'])].sort_values('origin').copy()   # every catalog detection of the window
    tref=pd.Timestamp(str(sh['origin_utc'][ti])[:23])
    for r in det.itertuples():
        m=relsize(ti,r.origin);g=m[(m.cc>=CC_MIN)&(m.snr_det>=SNR_MIN)&(m.snr_tpl>=SNR_MIN)&(m.ratio>0)]
        ev=mag[mag.event_id==r.reference_event_id] if r.is_reference else mag.iloc[0:0]      # the catalog's reference event, if any
        dM=float(np.median(np.log10(g.ratio))) if len(g)>=2 else np.nan;sd=float(1.4826*np.median(np.abs(np.log10(g.ratio)-dM))/np.sqrt(len(g))) if len(g)>=2 else np.nan
        allrows.append(dict(burst=B['name'],burst_key=B['key'],burst_template=B['tpl'],origin=r.origin,minutes_from_template=(r.origin-tref).total_seconds()/60,detected_by=int(r.template_cuspid),network_cc=float(r.cc),
            n_templates=int(r.n_templates),n_present=int(r.n_present),self_detection=bool(r.self_detection),is_reference=bool(r.is_reference),is_new=bool(r.is_new),
            best_channel_cc=float(m.cc.abs().max()) if len(m) else np.nan,
            n_ratio_channels=len(g),dM=dM,dM_se=sd,reference_event=str(ev.event_id.iloc[0]) if len(ev) else '',reference_ML=float(ev.ml_s.iloc[0]) if len(ev) and pd.notna(ev.ml_s.iloc[0]) else np.nan))
        for q in m.itertuples():chanrows.append(dict(burst=B['name'],origin=r.origin,station=q.station,comp=q.comp,cc=q.cc,lag_s=q.lag_s,ratio=q.ratio,snr_det=q.snr_det,snr_tpl=q.snr_tpl,used=(q.cc>=CC_MIN and q.snr_det>=SNR_MIN and q.snr_tpl>=SNR_MIN and q.ratio>0)))
    # repeater gather at the best station, every detection in the burst window, aligned on the burst template's picks
    grows=station_gather(B['tpl'],B['station'],B['t0'],B['t1'],os.path.join(OUT,f"G_{B['key']}.png"),title=f"{B['name']}: every detection at {B['station']}, aligned on template {B['tpl']}",same_template=False,dedup_s=0.0)
    for g in grows:GROWS.append(dict(burst=B['name'],station=B['station'],origin=g['origin'],detected_by=g['tpl'],network_cc=g['net_cc'],**{f'cc_{k}':v['cc'] for k,v in g['comp'].items()}))   # the correlations printed on the gather
    # EVERY detection except the burst template's own: record section + CC trace (the nb40 'set' figure), drawn with the template that
    # made the detection; a self-detection of another reference event is drawn with the burst template instead (to show the match)
    for r in det.itertuples():
        c=int(r.template_cuspid)
        if bool(r.self_detection) and c==B['tpl']:continue
        if bool(r.self_detection):c=B['tpl']
        k=pd.Timestamp(r.origin).strftime('%Y-%m-%d %H:%M:%S.%f')[:23]
        subprocess.run([sys.executable,os.path.join(HERE,'band_event_figs.py'),k,str(c)],check=True,capture_output=True,env=dict(os.environ,FMF_SET_ONLY='1'))
        stem=f"{pd.Timestamp(k):%Y%m%d_%H%M%S}_{c}";src=os.path.join(HERE,'figs_similarity_full_period_5_20H','new',stem+'_set.png')
        _link(src,os.path.join(OUT,f"S_{B['key']}_{pd.Timestamp(k):%Y%m%d_%H%M%S}.png"));SETS.append(dict(burst=B['name'],origin=r.origin,figure_template=c,png=f"S_{B['key']}_{pd.Timestamp(k):%Y%m%d_%H%M%S}.png"))
        print('set figure',k,c,flush=True)
t=pd.DataFrame(allrows)
# absolute ML per burst from anchors (detections that are reference events with an nb43 magnitude)
for b,g in t.groupby('burst'):
    a=g[g.reference_ML.notna()&g.dM.notna()]
    if len(a):
        mt=float(np.median(a.reference_ML-a.dM));t.loc[g.index,'template_ML']=mt;t.loc[g.index,'ML']=mt+g.dM;t.loc[g.index,'anchors']='; '.join(f"{x.reference_event} (ML {x.reference_ML:.2f})" for x in a.itertuples())
# day detection statistics of the figure template (logged by band_event_figs): network CC as a multiple of that day's MAD
lg=pd.read_csv(os.path.join(HERE,'logs','band_event_channels.csv'),parse_dates=['origin']);lg=lg[lg.run==C.FMF_RUN].drop_duplicates(['origin','template_cuspid'],keep='last')
sets=pd.DataFrame(SETS).merge(lg[['origin','template_cuspid','day_mad','day_thr','net_cc','n_live']].rename(columns={'template_cuspid':'figure_template'}),on=['origin','figure_template'],how='left')
sets.to_csv(os.path.join(OUT,'burst_set_figures.csv'),index=False)
for r in t.itertuples():
    m=lg[((lg.origin-r.origin).abs()<pd.Timedelta(seconds=0.5))&(lg.template_cuspid==r.detected_by)]
    if len(m):t.loc[r.Index,'day_mad']=float(m.day_mad.iloc[0]);t.loc[r.Index,'day_threshold']=float(m.day_thr.iloc[0])
t['cc_over_mad']=t.network_cc/t.get('day_mad',np.nan)
t.to_csv(os.path.join(OUT,'burst_detections.csv'),index=False);pd.DataFrame(chanrows).to_csv(os.path.join(OUT,'burst_ratio_channels.csv'),index=False)
pd.DataFrame(GROWS).to_csv(os.path.join(OUT,'burst_gather_cc.csv'),index=False)
print(t[[c for c in ['burst','origin','minutes_from_template','detected_by','network_cc','is_reference','best_channel_cc','n_ratio_channels','dM','dM_se','ML'] if c in t]].round(3).to_string(index=False))
