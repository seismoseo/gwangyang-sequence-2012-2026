"""Similarity figures and summary for the FMF detections (Buan-style): per-channel template-vs-detection overlays and the
one-template day CC trace for every NEW detection (figures + summary), and the summary (figures for the first --controls) for the
control population of real events detected by another event's template. Classes: final catalogue (decision 20, 2026-10-06):
control = a reference event not detected by its own template, new = not a reference event; as-run catalogues (2-15 Hz and mixed
5-20 Hz runs): control = in-DL non-self detection, new = not in the DL catalog. Self-detections are skipped.
Resume: rows of an existing summary are reused only when every one of them has the current class and label; a summary written for
another catalogue or label set stops the tool (move it, and the figures its 'figure' column names, aside first).
Outputs: fmf/figs_similarity/{new,control}/<YYYYmmdd_HHMMSS>_<cuspid>_{overlay,cc}.png and fmf/logs/similarity_summary.csv.
Per detection: recomputed network CC (scan weights), mean per-channel CC, channels read, channels with CC >= 0.5, whole-day
12xMAD threshold (the catalog used per-segment thresholds; 'passes_wholeday' says whether the detection clears the whole-day one),
the day trace's peak within +-12 s and the template's second-highest peak elsewhere in the day (prominence).
usage (scan environment): python make_similarity_figs.py [--controls N] [--limit N]
"""
import os,sys,time,argparse
import numpy as np,pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import config as C
import similarity as S
from obspy import UTCDateTime
ROOT=os.path.dirname(os.path.abspath(__file__));SUF='' if C.FMF_RUN in ('','full_period') else f'_{C.FMF_RUN}';FIG=os.path.join(ROOT,f'figs_similarity{SUF}');LOG=os.path.join(ROOT,'logs',f'similarity_summary{SUF}.csv')
ap=argparse.ArgumentParser();ap.add_argument('--controls',type=int,default=40,help='control detections that also get figures (all get summary rows)');ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
cat=pd.read_csv(C.CATALOG_CSV);ver=pd.read_csv(os.path.join(ROOT,'logs',f'candidate_verdicts{SUF}.csv')) if os.path.exists(os.path.join(ROOT,'logs',f'candidate_verdicts{SUF}.csv')) else None
FINAL='is_reference' in cat;known=cat.is_reference if FINAL else cat.in_DL
cat['cls']=np.where(cat.self_detection,'self',np.where(cat.is_new,'new',np.where(known,'control','in_template')))
# factual figure labels (2026-10-06, user rule: the pick/waveform triage words are internal, never in a figure)
LAB={'new':'new detection (not a reference event)' if FINAL else 'not in the DL catalog',
     'control':"reference event detected by another event's template" if FINAL else "in the DL catalog, detected by another event's template"}
sel=cat[cat.cls.isin(['new','control'])].copy()
if ver is not None:sel=sel.merge(ver[['origin','verdict','matched_stations']],on='origin',how='left')
CUR={(o,int(t)):(c,LAB[c]) for o,t,c in zip(sel.origin,sel.template_cuspid,sel.cls)}   # before --limit: the whole current selection
if a.limit:sel=sel.head(a.limit)
sel['day']=pd.to_datetime(sel.origin).dt.strftime('%Y-%j');sh=S.load_shelly();cusp=list(sh['cuspid'].astype(int))
for d in ('new','control'):os.makedirs(os.path.join(FIG,d),exist_ok=True)
rows=[];done=set()
if os.path.exists(LOG):
    prev=pd.read_csv(LOG);plab=prev['label'] if 'label' in prev else [None]*len(prev)
    stale=[k for k in zip(prev.origin,prev.template_cuspid.astype(int),prev.cls,plab) if CUR.get(k[:2])!=k[2:]]
    if stale:raise SystemExit(f"{LOG}: {len(stale)} of {len(prev)} rows have another class or label than the current catalogue (first: {stale[0]}); "
                              "move it, and the figures its 'figure' column names, aside before a fresh run")
    rows=prev.to_dict('records');done={(r['origin'],int(r['template_cuspid'])) for r in rows}
ctrl_figs=0;t_start=time.time()
for day,g in sel.groupby('day',sort=True):
    todo=[r for r in g.itertuples() if (r.origin,int(r.template_cuspid)) not in done]
    if not todo:continue
    y,doy=int(day[:4]),int(day[5:]);stations=[str(s) for s in sh['stations']];prefixes=[str(p) for p in sh['channels']]
    day=S.scan.load_day(stations,prefixes,y,doy);present=day[1]                    # the day's data loaded once for every detection of the day
    for r in todo:
        ti=cusp.index(int(r.template_cuspid));t0=time.time()
        try:
            ch,net_cc,mean_cc=S.detection_vs_template(sh,ti,r.origin,present=present)
            res=S.template_cc_day(sh,ti,y,doy,day=day)
            i,j,pk=S.cc_at(res,sh,ti,r.origin) if res is not None else (None,None,np.nan)
            if res is not None:
                cc=res['cc'];h=int(4*C.FS/C.STEP);m=cc.copy();m[max(0,i-h):i+h+1]=-np.inf;second=float(m.max())
            else:second=np.nan
            top=sorted([x['cc'] for x in ch],reverse=True)
            row=dict(origin=r.origin,year=r.year,template_cuspid=int(r.template_cuspid),cls=r.cls,label=LAB[r.cls],verdict=getattr(r,'verdict',''),catalog_cc=r.cc,n_present=r.n_present,n_templates=r.n_templates,
                     net_cc=net_cc,mean_channel_cc=mean_cc,n_channels=len(ch),n_channels_ge05=int(sum(c>=0.5 for c in top)),top1=top[0] if top else np.nan,top3_mean=float(np.mean(top[:3])) if top else np.nan,
                     wholeday_thr=res['thr'] if res else np.nan,day_mad=res['mad'] if res else np.nan,passes_wholeday=bool(res and pk>=res['thr']),peak_cc=pk,peak_offset_s=((j-i)*C.STEP/C.FS) if res else np.nan,
                     second_peak_day=second,prominence=(pk/max(second,1e-6)) if res else np.nan,n_live_channels=res['n_channels'] if res else 0)
            want_fig=(r.cls=='new') or (ctrl_figs<a.controls)
            if want_fig and ch:
                stem=os.path.join(FIG,r.cls,f"{pd.Timestamp(r.origin):%Y%m%d_%H%M%S}_{int(r.template_cuspid)}")
                lab=LAB[r.cls]                                 # factual label only (see LAB)
                # never overwrite a figure another tool already wrote under the same name (band_event_figs.py uses the same
                # plotting functions with the band in the title, and nb40 reads those files): this tool then writes <stem>_sim_*
                if os.path.exists(stem+'_overlay.png') or os.path.exists(stem+'_cc.png'):stem=stem+'_sim'
                S.plot_overlay(ch,f"Template {int(r.template_cuspid)} vs detected waveform ({pd.Timestamp(r.origin):%Y-%m-%d %H:%M:%S} UTC, network CC {net_cc:.2f}, catalog CC {r.cc:.2f}) — {lab}",stem+'_overlay.png')
                if res is not None:S.plot_cc_trace(res,sh,ti,r.origin,f"{pd.Timestamp(r.origin):%Y-%m-%d %H:%M:%S} UTC | template {int(r.template_cuspid)} | catalog CC {r.cc:.2f} | {r.n_present} present | {r.n_templates} templates | {lab}",stem+'_cc.png')
                row['figure']=os.path.relpath(stem,ROOT)
                if r.cls=='control':ctrl_figs+=1
            rows.append(row);done.add((r.origin,int(r.template_cuspid)))
            print(f"{r.origin[:19]} tpl {int(r.template_cuspid)} {r.cls:7s} cat {r.cc:.2f} net {net_cc:.2f} mean {mean_cc:.2f} ch {len(ch)} peak {pk:.2f} thr {row['wholeday_thr']:.2f} prom {row['prominence']:.1f} | {time.time()-t0:.0f} s",flush=True)
        except Exception as e:
            print(f"!! {r.origin} tpl {r.template_cuspid}: {e}",flush=True)
    pd.DataFrame(rows).to_csv(LOG,index=False);del day
print(f"done {len(rows)} rows in {(time.time()-t_start)/60:.1f} min -> {LOG}")
