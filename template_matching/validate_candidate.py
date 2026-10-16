"""Pick-based evidence for an FMF candidate: unassociated PhaseNet picks of that day at the template's stations within
-60/+90 s of the detection, and observed vs template S-P where a station has both picks (the nb37 validation recipe).
usage: python validate_candidate.py 'YYYY-MM-DD HH:MM:SS.fff' TEMPLATE_CUSPID   (prints one line per pick and the S-P comparison)
       python validate_candidate.py --all   (every declustered detection of the FMF catalog -> fmf/logs/candidate_pick_support.csv)
A pick "matches" when it lies within TOL of the arrival predicted from the template (detection origin + ref_s + moveout).
"""
import os,sys
from pathlib import Path
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent                # the template-matching folder
import pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # registry paths
RUN=os.environ.get('FMF_RUN','full_period');SEG='_seg' if os.environ.get('MAD_SEGMENTED','1') not in ('','0','false','False') else '';SUF='' if RUN=='full_period' else f'_{RUN}'
SHELLY=HERE/f'templates_cache/haenam_shelly_{RUN}.npz';CATALOG=HERE/f'catalogs/fmf_haenam_catalog_{RUN}{SEG}.csv'
TOL=0.3                                              # s: a pick within this of the template-predicted arrival counts as support (disclosed)
WF_CC=0.7                                            # per-channel template CC that counts as waveform support at a station (controls: 60 % of real events reach it at >= 2 stations)
PRE_PICK=0.50                                        # s: arrival = detection origin + ref_s + PRE_PICK + moveout; calibrated on the 4375 picks of the self-detections (median offset 0.503 s, MAD 0.005 s)
_SH=None;_PICKS={}
# record sections inspected by eye (nb37 R?? figures), keyed by origin to the millisecond
RECORD_SECTION_NOTES={'2023-06-17 19:30:26.856':'R: S bursts at the predicted S on JDO2/AJD/MOPB/HAWA','2023-01-12 07:43:21.977':'R: no P/S energy at the predicted times',
 '2024-04-21 02:07:32.687':'R: no P/S energy at the predicted times; coincident non-local picks at YEGA/JAHA/NHPA and, 42 s later, GKSA/NYSA/GUNA',
 '2020-10-19 18:53:33.787':'R: impulsive S at the predicted S on JDO2 and AJD, P on AJD','2020-10-16 20:29:11.167':'R: S at the predicted S on JDO2 and AJD','2026-04-29 18:47:09.490':'R: impulsive P at the predicted P on JN01, S at the predicted S on JDO2 and JN08'}
def _shelly():
    global _SH
    if _SH is None:
        sh=np.load(SHELLY,allow_pickle=True);_SH=dict(st=[str(s) for s in sh['stations']],cusp=list(sh['cuspid'].astype(int)),Pm=sh['P_moveouts'][:,:,0],Sm=sh['S_moveouts'][:,:,0],Pw=sh['P_weights'][:,:,0],Sw=sh['S_weights'][:,:,0],ref=sh['ref_s'])
    return _SH
def _picks(y,d):
    if (y,d) not in _PICKS:
        f=haenam.ROOT / 'detection/phasenet_plus_2020-2026/detection_location' / f'{y}/picks/picks_{y}.{d:03d}.csv'
        p=pd.read_csv(f) if f.exists() else pd.DataFrame(columns=['station','phase','peak_time','probability']);p['t']=pd.to_datetime(p.peak_time);_PICKS[(y,d)]=p
    return _PICKS[(y,d)]
def evidence(origin,cuspid,pre=60,post=90):
    # predicted absolute arrivals: detection origin + ref_s (origin -> start of the earliest template window) + PRE_PICK (window
    # starts PRE_PICK s before the pick) + moveout/100 Hz; calibrated on self-detections (residual 0.00-0.01 s)
    S=_shelly();st=S['st'];i=S['cusp'].index(int(cuspid));Pm,Sm,Pw,Sw,ref=S['Pm'][i],S['Sm'][i],S['Pw'][i],S['Sw'][i],float(S['ref'][i])+PRE_PICK
    tpl=[s for k,s in enumerate(st) if Pw[k]>0 or Sw[k]>0];sp_tpl={s:(Sm[k]-Pm[k])/100 for k,s in enumerate(st) if Pw[k]>0 and Sw[k]>0}
    pred={(s,'P'):ref+Pm[k]/100 for k,s in enumerate(st) if Pw[k]>0};pred.update({(s,'S'):ref+Sm[k]/100 for k,s in enumerate(st) if Sw[k]>0})
    det=pd.Timestamp(origin);y=det.year;d=det.dayofyear;p=_picks(y,d)
    w=p[(p.t>det-pd.Timedelta(seconds=pre))&(p.t<det+pd.Timedelta(seconds=post))];wt=w[w.station.isin(tpl)].sort_values('t')
    picks=[f"{r.station} {r.phase} {(r.t-det).total_seconds():+.2f}s p{r.probability:.2f}" for r in wt.itertuples()]
    matched=[];res=[]
    for r in wt.itertuples():
        key=(r.station,r.phase)
        if key in pred:
            dt=(r.t-det).total_seconds()-pred[key]
            if abs(dt)<=TOL:matched.append(f'{r.station} {r.phase} {dt:+.2f}');res.append(dt)
    sp=[]
    for s,g in wt.groupby('station'):
        if {'P','S'}<=set(g.phase) and s in sp_tpl:sp.append(f"{s} observed {(g[g.phase=='S'].t.min()-g[g.phase=='P'].t.min()).total_seconds():.2f} s vs template {sp_tpl[s]:.2f} s")
    return dict(origin=str(origin),template_cuspid=int(cuspid),template_stations=len(tpl),picks_at_template_stations=len(wt),picks=';'.join(picks) or 'none',
                matched_phases=len(matched),matched_stations=len({m.split()[0] for m in matched}),matched=';'.join(matched) or 'none',matched_rms_s=float(np.sqrt(np.mean(np.square(res)))) if res else np.nan,
                s_minus_p=' | '.join(sp) or 'no station with both P and S',other_picks_in_window=int(len(w)-len(wt)))
if __name__=='__main__':
    if sys.argv[1]=='--all':
        c=pd.read_csv(CATALOG);rows=[]
        # class columns carried per detection: the final catalogue's (decision 20, 2026-10-06: reference event or new), or in_DL for
        # an as-run catalogue (the 2-15 Hz and mixed 5-20 Hz runs keep the as-run columns)
        cls=(lambda r:dict(is_reference=r.is_reference,reference_event_id=r.reference_event_id)) if 'is_reference' in c else (lambda r:dict(in_DL=r.in_DL))
        for r in c.itertuples():
            e=evidence(r.origin,r.template_cuspid);e.update(year=r.year,cc=r.cc,n_present=r.n_present,n_templates=r.n_templates,self_detection=r.self_detection,**cls(r),is_new=r.is_new);rows.append(e)
        out=pd.DataFrame(rows);out['tier']=np.where(out.cc>=0.6,'A',np.where((out.cc>=0.4)&(out.n_templates>=3),'B','C'))
        # verdict rule (disclosed): validated = picks within TOL of the predicted arrival at >= 2 template stations; pick-supported = 1 phase; rejected = none
        out['verdict']=np.select([out.matched_stations>=2,out.matched_phases>=1],['validated','pick-supported (1 phase)'],'rejected')
        out['record_section']=[RECORD_SECTION_NOTES.get(o[:23],'') for o in out.origin]
        # waveform support from the per-channel template correlations (fmf/channel_cc.py -> logs/similarity_channels.csv):
        # stations with a channel CC >= WF_CC; combined verdict (disclosed): validated = picks at >= 2 stations OR waveform at >= 2
        # stations; supported = one matched phase OR one waveform station; unsupported = neither
        chf=HERE/f'logs/similarity_channels{SUF}.csv'
        if chf.exists():
            ch=pd.read_csv(chf);g=ch.groupby('origin');wf=pd.DataFrame(dict(waveform_stations_ge07=g.apply(lambda d:d[d.cc>=WF_CC].station.nunique()),waveform_stations_ge06=g.apply(lambda d:d[d.cc>=0.6].station.nunique()),waveform_best_cc=g.cc.max(),waveform_channels=g.size())).reset_index()
            out=out.merge(wf,on='origin',how='left')
        else:
            out['waveform_stations_ge07']=np.nan;out['waveform_stations_ge06']=np.nan;out['waveform_best_cc']=np.nan;out['waveform_channels']=np.nan
        w7=out.waveform_stations_ge07.fillna(0)
        out['final_verdict']=np.select([(out.matched_stations>=2)|(w7>=2),(out.matched_phases>=1)|(w7>=1)],['validated','supported'],'unsupported')
        out.to_csv(HERE/f'logs/candidate_pick_support{SUF}.csv',index=False);print('wrote',len(out),'rows ->',HERE/f'logs/candidate_pick_support{SUF}.csv')
        v=out[out.is_new][['origin','year','template_cuspid','cc','n_present','n_templates','tier','template_stations','picks_at_template_stations','matched_phases','matched_stations','matched_rms_s','matched','s_minus_p','verdict','waveform_stations_ge07','waveform_stations_ge06','waveform_best_cc','waveform_channels','final_verdict','record_section']]
        v.to_csv(HERE/f'logs/candidate_verdicts{SUF}.csv',index=False);print('wrote',len(v),'new candidates ->',HERE/f'logs/candidate_verdicts{SUF}.csv');print('pick verdict:',v.verdict.value_counts().to_dict());print('final verdict:',v.final_verdict.value_counts().to_dict())
    else:
        for k,v in evidence(' '.join(sys.argv[1:-1]),sys.argv[-1]).items():print(f'{k}: {v}')
