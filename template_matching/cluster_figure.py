"""One figure for a short-lived cluster: every detection of the cluster on ONE continuous record window, with the template's
network-CC trace over exactly the same window underneath, so the waveforms and the detector's view share a time axis.

usage (run env): python cluster_figure.py CUSPID 'START_UTC' 'END_UTC' OUTFILE ['TITLE']
Rows are station-components ordered by distance from the template event; for each detection in the window the template's P
(red) and S (blue) windows are drawn in place on the record, so a repeat shows as the same pattern recurring at each
detection time. The CC panel marks every detection of the catalog and the detection threshold of that day.
Detection labels: final catalogue (decision 20, 2026-10-06) template event / reference event / new; an as-run catalogue (2-15 Hz
run) keeps its own labels (in / not in the DL catalog). The as-run copy-rule marking ('removed: copy', from *_copies_removed.csv)
was dropped with the copy rule; the version that drew it is kept in the cleanup backups (docs/cleanup/fmf_final_backups).
"""
import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # the project registry, found upward from this file
import os,sys
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import config as C, similarity as S
HERE=os.path.dirname(os.path.abspath(__file__))
NMAX=int(os.environ.get("FMF_CLUSTER_NMAX","0"))       # station-components drawn (nearest first); 0 = all live channels
FARTHEST=os.environ.get('FMF_CLUSTER_FARTHEST','SIJA.Z')  # keep rows out to this station-component (by distance); '' = no limit
                                                          # (beyond ~31 km the late arrivals of the last event leave the window)

def _label(r,final):
    """Factual label of one catalogue detection: final catalogue columns (decision 20) or the as-run ones."""
    if r.self_detection:return 'template event'
    if final:return 'reference event' if r.is_reference else 'new'
    return 'in the DL catalog' if r.in_DL else ('reference event' if r.in_template else 'not in the DL catalog')

def cluster_figure(cuspid,t0,t1,outfile,title=None,nmax=NMAX,focus=None):
    """focus: an origin time; if given, every trace is scaled to its own maximum inside the span of the template windows placed at that
    detection (clipped at 1.5x), so weak arrivals there stay visible next to a large event; default (None) scales to the whole window."""
    sh=S.load_shelly();cusp=list(sh['cuspid'].astype(int));ti=cusp.index(int(cuspid))
    stations=[str(x) for x in sh['stations']];prefixes=[str(p) for p in sh['channels']];comps=[str(c) for c in sh['components']]
    tpl=pd.read_csv(C.TEMPLATE_CSV)   # the run's own template table.set_index('cuspid')
    row=tpl.loc[int(cuspid)];lat,lon=float(row.latitude),float(row.longitude)
    coords=S._station_coords()
    T0,T1=UTCDateTime(t0),UTCDateTime(t1)
    cat=pd.read_csv(C.CATALOG_CSV,parse_dates=['origin'])
    det=cat[(cat.origin>=pd.Timestamp(t0))&(cat.origin<=pd.Timestamp(t1))].copy().sort_values('origin')
    final='is_reference' in cat
    # day CC trace of the template (the detector's own view), cut to the same window
    day=S.scan.load_day(stations,prefixes,T0.year,T0.julday)
    res=S.template_cc_day(sh,ti,T0.year,T0.julday,day=day)
    # rows: the template's live channels, nearest first
    ch=[]
    for si,ph,ci,mo,n,tw,j in S.channels_of(sh,ti):
        name=stations[si];km=gps2dist_azimuth(lat,lon,*coords[name])[0]/1000.0 if name in coords else np.nan
        ch.append(dict(si=si,name=name,comp=comps[ci],phase=ph,mo=mo,n=n,tw=tw,km=km))
    by={}
    for c in ch:by.setdefault((c['name'],c['comp']),{})[c['phase']]=c
    keys=sorted(by,key=lambda k:(np.nan_to_num(by[k][list(by[k])[0]]['km'],nan=9e9),k))
    if FARTHEST:                                   # keep every channel of every station no farther than FARTHEST's station
        want=FARTHEST.split('.')[0]
        lim=next((by[k][list(by[k])[0]]['km'] for k in keys if k[0].split('.')[1]==want),None)
        if lim is not None:keys=[k for k in keys if np.nan_to_num(by[k][list(by[k])[0]]['km'],nan=9e9)<=lim+1e-6]
    if nmax:keys=keys[:nmax]
    fig=plt.figure(figsize=(13.5,0.48*len(keys)+3.0))
    gs=fig.add_gridspec(2,1,height_ratios=[0.48*len(keys)+0.5,2.0],hspace=0.16,top=0.955,bottom=0.07)
    ax=fig.add_subplot(gs[0]);axc=fig.add_subplot(gs[1],sharex=ax)
    rel=lambda t:float(UTCDateTime(t)-T0)
    for k,key in enumerate(keys):
        name,comp=key;net,sta=name.split('.')
        d=S.read_window(net,sta,comp,prefixes[stations.index(name)],T0,int(round((T1-T0)*C.FS)))
        if d is None:continue
        t=np.arange(d.size)/C.FS;a=np.abs(d).max() or 1
        if focus is not None:                            # scale to the record inside this channel's windows at the focus detection
            fo=UTCDateTime(pd.Timestamp(focus).to_pydatetime());sp=[]
            for phase in ('P','S'):
                c_=by[key].get(phase)
                if c_ is not None:st_=rel(fo+float(sh['ref_s'][ti])+c_['mo']/C.FS);sp+=[st_,st_+c_['n']/C.FS]
            if sp:
                m_=(t>=min(sp))&(t<=max(sp));a=(np.abs(d[m_]).max() if m_.any() else a) or a
            d=np.clip(d,-1.5*a,1.5*a)
        ax.plot(t,0.42*d/a+k,color='.55',lw=.5,zorder=1)
        for r in det.itertuples():                       # every detection: draw the template windows where they fall
            o=UTCDateTime(pd.Timestamp(r.origin).to_pydatetime())
            for phase,colr in (('P','#c1352b'),('S','#1b3fa8')):
                c=by[key].get(phase)
                if c is None:continue
                st=rel(o+float(sh['ref_s'][ti])+c['mo']/C.FS);tt=st+np.arange(c['n'])/C.FS
                m=(t>=tt[0])&(t<=tt[-1]);amp=np.abs(d[m]).max()/a if m.any() else 1.0
                ax.plot(tt,c['tw']/(np.abs(c['tw']).max() or 1)*amp*0.42+k,color=colr,lw=1.0,zorder=3)
                ax.plot([tt[0]+0.5]*2,[k-.45,k+.45],color=colr,lw=.8,ls=(0,(3,2)),zorder=4)
        ax.text(-0.008,k,f"{sta}.{comp}"+(f"  {by[key][list(by[key])[0]]['km']:.0f} km" if np.isfinite(by[key][list(by[key])[0]]['km']) else ''),
                transform=ax.get_yaxis_transform(),fontsize=7.5,ha='right',va='center')
    ax.set_yticks([]);ax.set_ylim(-.8,len(keys)-.2)
    ax.text(.004,1.008,'record (gray'+(', each trace scaled inside the detection\u2019s windows and clipped at 1.5\u00d7' if focus is not None else '')+') · template P window (red) · template S window (blue) · dashed: the picks',transform=ax.transAxes,fontsize=7.5,va='bottom')
    if res is not None:
        i0=int(round((T0-res['day0'])*C.FS/C.STEP));i1=int(round((T1-res['day0'])*C.FS/C.STEP))
        i0=max(0,i0);i1=min(res['cc'].size,i1);tt=(np.arange(i0,i1)-i0)*C.STEP/C.FS
        axc.plot(tt,res['cc'][i0:i1],color='#1b3fa8',lw=.7)
        axc.axhline(res['thr'],color='#2a9d3f',lw=1,label=f"12×MAD = {res['thr']:.2f}");axc.axhline(res['floor'],color='#e08a1e',lw=1,ls='--',label=f"floor {res['floor']:.1f}")
        axc.legend(fontsize=7.5,loc='upper left',frameon=True,framealpha=1).set_zorder(10)
    # mark each detection where the DETECTOR places it: the CC peak (= origin + ref_s), which is also where the first
    # template window starts on the record, so the marker lines up with the waveforms above it
    for r in det.itertuples():
        x=rel(UTCDateTime(pd.Timestamp(r.origin).to_pydatetime())+float(sh['ref_s'][ti]))
        lab=_label(r,final);col='#c1352b' if r.self_detection else '#1b3fa8'
        for a in (ax,axc):a.axvline(x,color=col,lw=.7,ls=':',alpha=.7,zorder=0)
        axc.annotate(f"{pd.Timestamp(r.origin):%H:%M:%S}  CC {r.cc:.2f}\n{lab}",(x,1.03),xycoords=('data','axes fraction'),ha='center',va='bottom',fontsize=7,color=col)
    axc.set(xlabel=f'Seconds from {pd.Timestamp(t0):%Y-%m-%d %H:%M:%S} UTC',ylabel='Network CC',xlim=(0,float(T1-T0)))
    axc.margins(y=.35)
    fig.suptitle(title or f'Cluster around template {cuspid}',fontsize=11,y=0.995)
    fig.savefig(outfile,dpi=150,bbox_inches='tight',pad_inches=0.15);plt.close(fig)
    print(f'{len(det)} detections in the window -> {outfile}')
    return det

if __name__=='__main__':
    cluster_figure(int(sys.argv[1]),sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5] if len(sys.argv)>5 else None)
