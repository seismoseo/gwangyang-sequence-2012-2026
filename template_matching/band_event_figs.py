"""Similarity figures + per-channel table for chosen detections in ONE run (env FMF_RUN/FMF_BAND decide the run), so the
same event can be compared across bands. usage (scan env): python band_event_figs.py 'ORIGIN' CUSPID [CUSPID ...]
Writes figs_similarity[_<run>]/new/<stamp>_<cuspid>_{overlay,cc}.png and appends rows to logs/band_event_channels.csv."""
import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # the project registry, found upward from this file
import os,sys
import numpy as np,pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import config as C,similarity as S
HERE=os.path.dirname(os.path.abspath(__file__));RUN=C.FMF_RUN or 'full_period';SUF='' if RUN=='full_period' else f'_{RUN}';FIG=os.path.join(HERE,f'figs_similarity{SUF}','new');os.makedirs(FIG,exist_ok=True)
LOG=os.path.join(HERE,'logs','band_event_channels.csv');origin=sys.argv[1];cusps=[int(x) for x in sys.argv[2:]]
sh=S.load_shelly();cusp=list(sh['cuspid'].astype(int));det=pd.Timestamp(origin)
tplcat=pd.read_csv(C.TEMPLATE_CSV).set_index('cuspid');stations=[str(s) for s in sh['stations']];prefixes=[str(p) for p in sh['channels']]
day=S.scan.load_day(stations,prefixes,det.year,det.dayofyear);present=day[1];rows=[]
for c in cusps:
    ti=cusp.index(c);ch,net,mean=S.detection_vs_template(sh,ti,origin,present=present);res=S.template_cc_day(sh,ti,det.year,det.dayofyear,day=day)
    i,j,pk=S.cc_at(res,sh,ti,origin) if res is not None else (None,None,np.nan)
    stem=os.path.join(FIG,f"{det:%Y%m%d_%H%M%S}_{c}");band=f'{C.BANDPASS[0]:g}–{C.BANDPASS[1]:g} Hz'
    tt=tplcat.loc[c] if c in tplcat.index else None
    if os.environ.get('FMF_ALL_FIGS'):        # the older per-detection views, kept behind a flag
        S.plot_overlay(ch,f'Template {c} vs detected waveform ({det:%Y-%m-%d %H:%M:%S} UTC, {band}, network CC {net:.2f})',stem+'_overlay.png')
        if res is not None:S.plot_cc_trace(res,sh,ti,origin,f'{det:%Y-%m-%d %H:%M:%S} UTC | template {c} | {band} | network CC {net:.2f}',stem+'_cc.png')
        S.plot_station_overlay(sh,ti,origin,ch,f'Template {c} P and S windows, {det:%Y-%m-%d %H:%M:%S} UTC',stem+'_station.png')
    if not os.environ.get('FMF_SET_ONLY'):S.plot_full_trace_comparison(sh,ti,origin,f'Full waveforms: template {c} (black) and the detection of {det:%Y-%m-%d %H:%M:%S} UTC (red), {band}',
                                 stem+'_full.png',tpl_lat=float(tt.latitude) if tt is not None else None,tpl_lon=float(tt.longitude) if tt is not None else None)
    if not os.environ.get('FMF_SET_ONLY'):S.plot_full_trace_grid(sh,ti,origin,f'Full waveforms by component: template {c} (black) and the detection of {det:%Y-%m-%d %H:%M:%S} UTC (red), {band}',
                           stem+'_grid.png',tpl_lat=float(tt.latitude) if tt is not None else None,tpl_lon=float(tt.longitude) if tt is not None else None)
    S.plot_detection_set(sh,ti,origin,ch,f'{det:%Y-%m-%d %H:%M:%S} UTC  ·  template {c}  ·  {band}  ·  network CC {net:.2f}',stem+'_set.png',
                         res=res,tpl_lat=float(tt.latitude) if tt is not None else None,tpl_lon=float(tt.longitude) if tt is not None else None)
    for o in ch:rows.append(dict(run=RUN,band=band,origin=origin,template_cuspid=c,station=o['station'],comp=o['comp'],phase=o['phase'],cc=o['cc'],weight=o['weight'],net_cc=net,mean_cc=mean,day_thr=res['thr'] if res else np.nan,day_mad=res['mad'] if res else np.nan,peak_cc=pk,n_live=res['n_channels'] if res else 0))
    print(f"{RUN} {band} tpl {c}: {len(ch)} ch ({sum(o['phase']=='P' for o in ch)} P + {sum(o['phase']=='S' for o in ch)} S), net {net:.3f}, mean {mean:.3f}, day thr {res['thr'] if res else float('nan'):.3f}, peak {pk:.3f}",flush=True)
new=pd.DataFrame(rows)
if os.path.exists(LOG):
    old=pd.read_csv(LOG);old=old[~((old.run==RUN)&(old.origin==origin)&old.template_cuspid.isin(cusps))];new=pd.concat([old,new],ignore_index=True)
new.to_csv(LOG,index=False);print('->',LOG,len(new),'rows')
