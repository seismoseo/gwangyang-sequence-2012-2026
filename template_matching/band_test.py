"""Band test for the FMF detections: re-cut the template's own windows and the detection's windows from the RAW continuous
data at several pass-bands and compare the per-channel CC (with a +-0.1 s lag search) and the detection's window SNR.
Window geometry as the scan: P [pick-0.5, pick+1.0] s on Z, S [pick-0.5, pick+1.5] s on the S component(s); picks of the
template event from the Shelly npz (origin + ref_s + 0.5 + moveout, calibrated on self-detections); the detection's picks
= detection origin + the same offsets. usage (scan env): python band_test.py  -> ../docs/history/template_matching/band_test.csv + figs_similarity/band_test_*.png
"""
import os,sys,glob
import numpy as np,pandas as pd
from obspy import read,UTCDateTime
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import config as C
import similarity as S
BANDS=[(2,8),(2,15),(4,16),(5,20),(8,25),(10,30)]
LAG=0.10                                              # s: lag search half-width per channel (the scan is zero-lag on the moveouts)
EVENTS=[('2023-09-15 01:05:58.307001',400177,'self-detection control'),('2020-10-19 18:53:33.787001',400139,'quiescence, validated'),('2020-10-16 20:29:11.167000',400146,'quiescence, validated'),
        ('2023-06-17 19:30:26.856999',400089,'quiescence, validated'),('2026-04-29 18:47:09.490000',400265,'quiescence, validated'),('2020-04-27 14:47:54.137001',400007,'2020 tier A'),
        ('2026-08-21 21:23:00.700001',400359,'2026 tier A'),('2024-04-21 02:07:32.687000',400031,'rejected'),('2023-01-12 07:43:21.977000',400171,'rejected')]
_raw={}
def raw(net,sta,comp,prefix,t0,t1):
    key=(net,sta,comp,t0.year,t0.julday)
    if key not in _raw:
        for pre in [prefix]+[q for q in ('HH','HG','EL') if q!=prefix]:
            f=glob.glob(f"{C.CONTINUOUS}/{net}/{sta}/{pre}{comp}.D/*.D.{t0.year}.{t0.julday:03d}")
            if f:break
        else:_raw[key]=None;return None
        try:
            st=read(f[0]);st.merge(method=1,fill_value=0);tr=st[0];tr.detrend('linear')
            if abs(tr.stats.sampling_rate-C.FS)>1e-6:tr.resample(C.FS)
            _raw[key]=tr
        except Exception:_raw[key]=None
    tr=_raw[key]
    return tr.slice(t0,t1) if tr is not None else None
def band_window(tr,band,t0,npts):
    x=tr.copy();x.filter('bandpass',freqmin=band[0],freqmax=band[1],corners=4,zerophase=True);seg=x.slice(t0,t0+(npts-1)/C.FS);d=seg.data.astype(np.float64)
    return d[:npts]-d[:npts].mean() if d.size>=npts else None
def cc_lag(a,b,maxlag):
    best=-2
    for L in range(-maxlag,maxlag+1):
        if L>=0:x,y=a[L:],b[:b.size-L]
        else:x,y=a[:a.size+L],b[-L:]
        den=np.sqrt((x**2).sum()*(y**2).sum());c=(x*y).sum()/den if den>0 else 0
        best=max(best,c)
    return best
sh=S.load_shelly();cusp=list(sh['cuspid'].astype(int));stations=[str(s) for s in sh['stations']];prefixes=[str(p) for p in sh['channels']];comps=[str(c) for c in sh['components']]
rows=[];ml=int(LAG*C.FS)
for o,c,lab in EVENTS:
    ti=cusp.index(c);torig=UTCDateTime(str(sh['origin_utc'][ti]));det=UTCDateTime(o);ref=float(sh['ref_s'][ti])
    for si,ph,ci,mo,n,tw,j in S.channels_of(sh,ti):
        net,sta=stations[si].split('.');pre=prefixes[si];comp=comps[ci];ws=ref+mo/C.FS      # window start rel. origin
        tt=raw(net,sta,comp,pre,torig+ws-30,torig+ws+n/C.FS+30);dd=raw(net,sta,comp,pre,det+ws-30,det+ws+n/C.FS+30)
        if tt is None or dd is None or tt.stats.npts<n+50*C.FS or dd.stats.npts<n+50*C.FS:continue
        for band in BANDS:
            a=band_window(tt,band,torig+ws,n);b=band_window(dd,band,det+ws,n)
            if a is None or b is None:continue
            noise=band_window(dd,band,det+ws-6.0,int(5*C.FS));snr=float(np.sqrt((b**2).mean())/max(np.sqrt((noise**2).mean()),1e-12)) if noise is not None else np.nan
            rows.append(dict(origin=o[:19],template_cuspid=c,label=lab,station=stations[si],comp=comp,phase=ph,band=f'{band[0]}-{band[1]}',cc0=float((a*b).sum()/max(np.sqrt((a**2).sum()*(b**2).sum()),1e-12)),cc_lag=cc_lag(a,b,ml),snr=snr))
    print(o[:19],lab,'channels',len({(r['station'],r['comp'],r['phase']) for r in rows if r['origin']==o[:19]}),flush=True)
d=pd.DataFrame(rows);d.to_csv(os.path.join(C.FMF_ROOT,'logs','band_test.csv'),index=False)
pd.set_option('display.width',250)
g=d.groupby(['label','origin','band']).agg(n=('cc0','size'),cc0_mean=('cc0','mean'),cclag_mean=('cc_lag','mean'),cclag_top3=('cc_lag',lambda x:np.sort(x)[-3:].mean()),n_ge05=('cc_lag',lambda x:(x>=.5).sum()),snr_med=('snr','median')).round(3)
print(g.to_string())
