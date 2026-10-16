"""Waveform figures of notebook 46 (the template-matching catalogue procedure, decision 20), drawn in the scan environment from
the inputs that notebooks/joint_fmf_counts.py writes (figs_pub/joint_fmf_counts/waveform_*.csv). Reads continuous data only.

  W01_six_template_records    each excluded template's own record: the continuous record through its correlation windows, from
                              3 s before the preceding DL event (same view as catalog_rules_review/bad_template_windows.py)
  W02_lost_<stamp>_<cuspid>   record section and the template's network CC over the day (similarity.plot_detection_set) for the
                              detections lost with the six templates that are not repeats at a built-in delay, and for the
                              repeat with the highest network CC; W02_checks.csv compares the recomputed network CC with the catalogue
  W03_near_large_events       vertical records at the three nearest stations for the as-run 20 s rule rows
  W04_closest_new             the same for the new detections closest to a reference origin

Run (from fmf/, only FMF_RUN exported; config.py loads the run's other settings):
    FMF_RUN=full_period_5_20H conda run -n fmf python catalog_procedure_figs.py
"""
import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # the project registry, found upward from this file
import os,sys
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from obspy import UTCDateTime
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,HERE)
import config as C,similarity as S
ROOT=os.path.dirname(HERE);OUT=os.environ.get('HAENAM_FMF46_OUTPUT',str(haenam.ROOT / 'analysis/seismicity/figs_pub/joint_fmf_counts'))
if C.FMF_RUN!='full_period_5_20H':raise SystemExit(f'FMF_RUN={C.FMF_RUN!r}; this script draws the full_period_5_20H run')
P_C,S_C,DET_C,OTH_C,PREV_C='#c1352b','#1b3fa8','#1b3fa8','#c1352b','#e07b39'
S._use_helvetica()
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42})
sh=S.load_shelly();cus=list(sh['cuspid'].astype(int));stations=[str(s) for s in sh['stations']];prefixes=[str(p) for p in sh['channels']];comps=[str(c) for c in sh['components']]
tc=pd.read_csv(haenam.path(None, 'tm_templates_594'),parse_dates=['origin_time_UTC']).set_index('cuspid')
dl=pd.read_csv(haenam.path(None, 'dl_catalog'),parse_dates=['time']).sort_values('time')
lost_cat=pd.read_csv(str(C.CATALOG_CSV)[:-len('.csv')]+'_excluded_template_detections.csv',parse_dates=['origin'])   # the build's list of lost detections (catalogue CC)
st=pd.read_csv(C.STATIONS_CSV);sensor={f'{r.Network}.{r.Code}':r.Sensor for r in st.itertuples()}
BAND=f'{C.BANDPASS[0]:g}–{C.BANDPASS[1]:g} Hz'

def legend_above(fig,handles,ncol,y=1.0):
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(0.5,y),ncol=ncol,frameon=True,framealpha=1.0,edgecolor='0.7',fontsize=8.5)

def own_records(templates):
    """W01: the six templates' own records (the view of catalog_rules_review/bad_template_windows.py, P red / S blue)."""
    fig,axs=plt.subplots(3,2,figsize=(14,16.5),layout='constrained')
    for ax,T in zip(axs.ravel(),templates):
        ti=cus.index(T);oT=UTCDateTime(str(sh['origin_utc'][ti]));oTp=tc.loc[T,'origin_time_UTC']
        prev=dl[dl.time<oTp-pd.Timedelta('0.5s')].iloc[-1];oP=UTCDateTime(prev.time.to_pydatetime())
        ch=sorted(S.channels_of(sh,ti),key=lambda c:(c[0],c[1]))[:10]
        wins=[(oT+float(sh['ref_s'][ti])+mo/C.FS,n,ph,ci,si,tw) for si,ph,ci,mo,n,tw,j in ch]
        t0=oP-3;t1=max(w[0]+w[1]/C.FS for w in wins)+3;npts=int((t1-t0)*C.FS);labels=[]
        for k,(ws,n,ph,ci,si,tw) in enumerate(wins):
            net,sta=stations[si].split('.');d=S.read_window(net,sta,comps[ci],prefixes[si],t0,npts);labels.append(f'{sta}.{comps[ci]} ({ph})')
            if d is None:continue
            x=np.arange(d.size)/C.FS;a=int((ws-t0-.5)*C.FS);b=int((ws-t0)*C.FS)+n+50
            sc=np.abs(d[max(a,0):b]).max() or 1;y=np.clip(d/sc,-1.3,1.3)*.42
            ax.plot(x,y+k,color='.35',lw=.35)
            col=P_C if ph=='P' else S_C;s0=ws-t0
            ax.fill_between([s0,s0+n/C.FS],k-.48,k+.48,color=col,alpha=.13,lw=0)
            twn=tw/(np.abs(tw).max() or 1)*.42*(np.abs(d[int(s0*C.FS):int(s0*C.FS)+n]).max()/sc)
            ax.plot(s0+np.arange(n)/C.FS,twn+k,color=col,lw=.7)
        ax.axvline(oP-t0,color=PREV_C,lw=1.1);ax.axvline(oT-t0,color='k',lw=1.1)
        ax.set_yticks(range(len(labels)),labels,fontsize=7.5);ax.set_xlim(0,t1-t0);ax.set_ylim(-.8,len(wins)-.2)
        ax.spines[['top','right']].set_visible(False);ax.set_xlabel('Seconds from 3 s before the preceding DL event')
        ax.set_title(f'Template {T} · {oTp:%Y-%m-%d %H:%M:%S} UTC · {(oTp-prev.time).total_seconds():.2f} s after a DL event with {int(prev.num)} picks',loc='left',fontsize=9)
    legend_above(fig,[Line2D([],[],color='.35',lw=.8,label=f'Record ({BAND}), each trace scaled inside its template window'),
                      Patch(color=P_C,alpha=.35,label='P window and template waveform'),Patch(color=S_C,alpha=.35,label='S window and template waveform'),
                      Line2D([],[],color=PREV_C,lw=1.2,label='Origin of the preceding DL event'),Line2D([],[],color='k',lw=1.2,label='Origin of the template event')],3)
    p=os.path.join(OUT,'W01_six_template_records.png');fig.savefig(p,dpi=110,bbox_inches='tight');plt.close(fig);print('wrote',p,flush=True)

def lost_sections(rows):
    """W02: record section + whole-day network CC of each listed detection (band_event_figs.py with FMF_SET_ONLY=1)."""
    checks=[];days={}
    for origin,T in rows:
        det=pd.Timestamp(origin);ti=cus.index(int(T));key=(det.year,det.dayofyear)
        if key not in days:days.clear();days[key]=S.scan.load_day(stations,prefixes,det.year,det.dayofyear)
        day=days[key];ch,net,mean=S.detection_vs_template(sh,ti,str(det),present=day[1]);res=S.template_cc_day(sh,ti,det.year,det.dayofyear,day=day)
        i,j,pk=S.cc_at(res,sh,ti,str(det)) if res is not None else (None,None,np.nan)
        tt=tc.loc[int(T)];p=os.path.join(OUT,f'W02_lost_{det:%Y%m%d_%H%M%S}_{int(T)}.png')
        k=int(np.argmin(np.abs((lost_cat.origin-det).dt.total_seconds().values)))
        if abs((lost_cat.origin.iloc[k]-det).total_seconds())>0.01 or int(lost_cat.template_cuspid.iloc[k])!=int(T):raise SystemExit(f'{det} {T} is not in the lost-detection list')
        ccat=float(lost_cat.cc.iloc[k])              # the title quotes the catalogue CC, as the notebook text does; W02_checks.csv adds the recomputed one
        S.plot_detection_set(sh,ti,str(det),ch,f'{det:%Y-%m-%d %H:%M:%S} UTC  ·  template {int(T)}  ·  {BAND}  ·  network CC {ccat:.2f}',p,res=res,
                             tpl_lat=float(tt.latitude),tpl_lon=float(tt.longitude))
        checks.append(dict(png=os.path.basename(p),origin=det,template=int(T),channels=len(ch),catalog_cc=ccat,network_cc_recomputed=net,day_trace_peak_cc=pk,
                           day_threshold=res['thr'] if res else np.nan))
        print('wrote',p,f'net CC {net:.3f}, day-trace peak {pk:.3f}',flush=True)
    pd.DataFrame(checks).to_csv(os.path.join(OUT,'W02_checks.csv'),index=False)

def section_panels(fig_id,rows,marks,nsta=3):
    """W03/W04: Z records at the nsta nearest stations (to the detecting template's event) with data; each trace scaled inside the
    detection's own predicted P-S window at that station (clipped at 1.5x), so a small detection next to a larger event is visible."""
    rows=rows[rows.figure==fig_id].reset_index(drop=True);n=len(rows);ncol=2 if n>4 else 1;nrow=int(np.ceil(n/ncol))
    fig,axs=plt.subplots(nrow,ncol,figsize=(7.4*ncol if ncol>1 else 11.0,2.25*nrow+.2),layout='constrained',squeeze=False)   # one column: wide enough for the legend
    for ax in axs.ravel()[n:]:ax.set_visible(False)
    for ax,r in zip(axs.ravel(),rows.itertuples()):
        det=UTCDateTime(r.detection.to_pydatetime());t0=UTCDateTime(r.t0.to_pydatetime());t1=UTCDateTime(r.t1.to_pydatetime());npts=int((t1-t0)*C.FS)
        mk=marks[marks.row==r.row];k=0;names=[]
        for item in str(r.stations).split(';'):
            name,km=item.split(':');net,sta=name.split('.')
            pre=prefixes[stations.index(name)] if name in stations else sensor.get(name,'HH')
            d=S.read_window(net,sta,'Z',pre,t0,npts)
            if d is None:continue
            x=np.arange(d.size)/C.FS+(t0-det);m=mk[mk.station==name]
            dp=m[(m.kind=='detection')];tp_=[(UTCDateTime(t.to_pydatetime())-det) for t in dp.time]
            lo,hi=(min(tp_)-.5,max(tp_)+1.5) if tp_ else (0.,10.);w=(x>=lo)&(x<=hi)
            sc=np.abs(d[w]).max() if w.any() else np.abs(d).max();y=np.clip(d/(sc or 1),-1.5,1.5)*.42
            ax.plot(x,y+k,color='.3',lw=.4)
            for q in m.itertuples():
                tq=UTCDateTime(q.time.to_pydatetime())-det
                if q.kind=='raw':ax.plot(tq,k+.5,marker='v',ms=4.5,color='k',mfc='k' if q.phase=='P' else 'white',mew=.8,zorder=5)
                else:ax.plot([tq,tq],[k-.45,k+.45],color=DET_C if q.kind=='detection' else OTH_C,lw=1.2,ls='-' if q.phase=='P' else (0,(3,1.5)),zorder=4)
            names.append(f'{sta} {float(km):.0f} km');k+=1
            if k==nsta:break
        dt=(r.detection-r.other_time).total_seconds()
        ax.set_yticks(range(len(names)),names,fontsize=7.5);ax.set_ylim(-.7,max(len(names),1)-.25);ax.set_xlim(t0-det,t1-det)
        ax.spines[['top','right']].set_visible(False)
        ax.set_title(f'{r.detection:%Y-%m-%d %H:%M:%S.%f}'[:-4]+f' · template {r.template} · CC {r.cc:.2f} · {r.n_templates} template{"s" if r.n_templates!=1 else ""} · '
                     f'{dt:+.2f} s from the {r.other_label}',loc='left',fontsize=8)
    for col in range(ncol):
        vis=[axs[i,col] for i in range(nrow) if i*ncol+col<n]
        if vis:vis[-1].set_xlabel('Seconds from the detection origin')
    other=rows.other_label.iloc[0]
    legend_above(fig,[Line2D([],[],color='.3',lw=.8,label=f'Z record ({BAND}), scaled inside the detection\'s predicted P–S window, clipped at 1.5×'),
                      Line2D([],[],color=DET_C,lw=1.2,label='Detection: predicted P'),Line2D([],[],color=DET_C,lw=1.2,ls=(0,(3,1.5)),label='Detection: predicted S'),
                      Line2D([],[],color=OTH_C,lw=1.2,label=f'The {other}: picked P'),Line2D([],[],color=OTH_C,lw=1.2,ls=(0,(3,1.5)),label=f'The {other}: picked S'),
                      Line2D([],[],marker='v',ls='',color='k',mfc='k',label='Raw PhaseNet+ P pick'),Line2D([],[],marker='v',ls='',color='k',mfc='white',label='Raw PhaseNet+ S pick')],4)
    p=os.path.join(OUT,f'{fig_id}_{"near_large_events" if fig_id=="W03" else "closest_new"}.png');fig.savefig(p,dpi=120,bbox_inches='tight');plt.close(fig);print('wrote',p,flush=True)

if __name__=='__main__':
    tg=pd.read_csv(os.path.join(OUT,'waveform_targets.csv'))
    rows=pd.read_csv(os.path.join(OUT,'waveform_rows.csv'),parse_dates=['detection','other_time','t0','t1'])
    marks=pd.read_csv(os.path.join(OUT,'waveform_marks.csv'),parse_dates=['time'])
    own_records([int(x) for x in tg[tg.kind=='own_record'].template])
    section_panels('W03',rows,marks);section_panels('W04',rows,marks)
    lost_sections([(o,int(T)) for o,T in zip(tg[tg.kind=='lost'].origin,tg[tg.kind=='lost'].template)])
