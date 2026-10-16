"""Relative local magnitude of an FMF detection, from EVERY template that detects it, with the nb34 calibration.

Measurement (per template j, per channel s of that template):
    r = <x_det, x_tpl> / <x_tpl, x_tpl>            least-squares scale of the detection window on the template window
    dML = beta_s * log10|r| + c_s                  nb34 calibration (5-20 Hz: beta 0.64-0.78 per station, NOT 1; the band sits
                                                   above the corner frequency of the larger anchors, so amplitude grows slower than ML)
    ML_det = ML_tpl + dML
Channel gates (nb34's safeguards carried over; all disclosed and testable via --sweep):
  * channel CC with the template >= CC_MIN               (an uncorrelated channel measures noise, not this event's amplitude)
  * detection-window SNR >= SNR_MIN                      (pre-arrival noise on the same channel, C.NOISE_WIN)
  * template-window SNR >= SNR_MIN                       (stored at build time in the Shelly npz)
  * same-era link only                                   (JDO2 changed gain between 2020 and 2026 without a metadata epoch)
Combination: median over channels within a template -> ML from that template; median over templates -> the event's ML.
Templates: every template with a raw hit within +-MATCH_S, except the six excluded by decision 20 (build_catalog.EXCLUDED_TEMPLATES).
Uncertainty: nb34 held-out rms for the link class, the inter-template scatter (robust sigma / sqrt(n_templates)) and the
inter-channel scatter added in quadrature; the inter-template scatter is also reported alone as an internal consistency check.
The calibration band is chosen from the run's own C.BANDPASS (2-15 Hz run -> nb34's 1-15 Hz calibration; 5-20 Hz run -> its
5-20 Hz calibration); mixing them is a systematic error of ~0.2 in beta and is refused.
usage (run env): python detection_magnitude.py 'ORIGIN' [CUSPID ...]   |   --catalog   |   --sweep   |   --selftest
"""
import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # the project registry, found upward from this file
import os,sys,json
import numpy as np,pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import config as C, similarity as S
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The calibration MUST match the band the waveforms are filtered in: nb34 fitted both, and beta differs (HAWA 0.78 at 5-20 Hz
# vs 0.93 at 1-15 Hz, with smaller residuals at 1-15). C.BANDPASS decides; a run whose band matches neither is refused.
_CAL_FOR_BAND={(5.0,20.0):'5-20',(2.0,15.0):'1-15',(1.0,15.0):'1-15'}
def _cal_name():
    b=(float(C.BANDPASS[0]),float(C.BANDPASS[1]))
    if b not in _CAL_FOR_BAND:
        raise SystemExit(f'no nb34 calibration for the scan band {b}; nb34 fitted 5-20 and 1-15 Hz (joint_relative_magnitude.BANDS)')
    return _CAL_FOR_BAND[b]
CAL_BAND=_cal_name()
CAL=os.path.join(str(haenam.ROOT / 'analysis/seismicity/figs_pub/joint_relative_magnitude'),f'calibration_{CAL_BAND}.csv')
HELD=str(haenam.ROOT / 'analysis/seismicity/figs_pub/joint_relative_magnitude/heldout_by_support.csv')
TPL=C.TEMPLATE_CSV   # the run's own template table (its FMF_TEMPLATE_CSV; R5: tm_templates_R4, ML = ml_s), so the cuspids are the scan's
# Gates chosen by --sweep on self-detections, whose true dML is 0 (logs/magnitude_sweep.log, 2026-09-28): the LS ratio is
# attenuated toward zero on poorly correlated / low-SNR channels (regression dilution), which biases ML DOWNWARD. Measured
# bias: -0.147 at CC>=0.3 with no SNR gate, -0.067 at CC>=0.5/SNR>=2, -0.016 at CC>=0.8/SNR>=3 (MAE 0.072). CC>=0.8 is also
# nb34's own anchor threshold, reached independently there.
CC_MIN=float(os.environ.get('FMF_MAG_CC_MIN','0.8'))     # channel CC gate
SNR_MIN=float(os.environ.get('FMF_MAG_SNR_MIN','3.0'))   # detection- and template-window SNR gate
MATCH_S=4.0                                              # a raw hit belongs to this event if within +-MATCH_S (the decluster window)
MIN_CHAN=2                                               # channels needed for one template's estimate
MAX_TEMPLATES=int(os.environ.get('FMF_MAG_MAX_TPL','30')) # templates per event, best CC first (all: ~8 h for 1305 events; 30
                                                          # reproduces the all-template median to <0.05 at a quarter of the cost)
DEFAULT_BETA=None                                        # set from the calibration actually loaded (median of its stations)
SAME_ERA_ONLY=True
_C=None
def _cal():
    global _C,DEFAULT_BETA
    if _C is None:
        c=pd.read_csv(CAL).set_index('station');_C=(c,pd.read_csv(HELD).set_index('support'))
        DEFAULT_BETA=float(c.beta.median())
    return _C
def _era(t):
    t=pd.Timestamp(t);return '2020' if t<pd.Timestamp('2020-08-01') else ('2026' if t>=pd.Timestamp('2026-08-14') else 'quiescence')
def _linkable(a,b):
    ea,eb=_era(a),_era(b);return (not SAME_ERA_ONLY) or ea==eb or 'quiescence' in (ea,eb)

def measure(sh,ti,det_origin,ml_tpl,tpl_origin,present=None):
    """Per-channel measurement for one (detection, template) link, WITHOUT gates: CC, both SNRs, the LS ratio and dML.
    Gates are applied by `gate_table`, so a sweep re-uses one measurement."""
    cal,_=_cal()
    if not np.isfinite(ml_tpl):return None,'template has no ML'
    if not _linkable(det_origin,tpl_origin):return None,'cross-era link'
    chans,_,_=S.detection_vs_template(sh,ti,det_origin,present=present)
    tsnr_P=sh['P_snr'][ti,:,0];tsnr_S=sh['S_snr'][ti];rows=[]
    for o in chans:
        ts=float(tsnr_P[o['si']]) if o['phase']=='P' else float(tsnr_S[o['si'],o['j']])
        t=o['template'];d=o['data'];den=float((t*t).sum());r=float((d*t).sum()/den) if den>0 else np.nan
        sta=o['station'].split('.')[1];b=float(cal.loc[sta,'beta']) if sta in cal.index else DEFAULT_BETA
        a=float(cal.loc[sta,'intercept']) if sta in cal.index else 0.0
        rows.append(dict(station=o['station'],comp=o['comp'],phase=o['phase'],cc=o['cc'],snr_det=o.get('snr',np.nan),snr_tpl=ts,
                         ratio=r,log10_ratio=np.log10(abs(r)) if (np.isfinite(r) and abs(r)>1e-12) else np.nan,beta=b,
                         calibrated=sta in cal.index,dML=b*np.log10(abs(r))+a if (np.isfinite(r) and abs(r)>1e-12) else np.nan))
    return pd.DataFrame(rows),''

def gate_table(t,cc_min,snr_min):
    """Apply the channel gates to a measurement table; returns the table with `used`/`gate` columns."""
    t=t.copy()
    t['gate']=np.where(t.cc<cc_min,'cc',np.where(~(t.snr_det>=snr_min),'snr_det',np.where(~(t.snr_tpl>=snr_min),'snr_tpl','')))
    t['used']=(t.gate=='')&t.dML.notna();return t

def template_estimate(sh,ti,det_origin,ml_tpl,tpl_origin,cc_min=CC_MIN,snr_min=SNR_MIN,present=None):
    """ML of the detection from ONE template, with the per-channel table. NaN when the link or the channels are unusable."""
    raw,reason=measure(sh,ti,det_origin,ml_tpl,tpl_origin,present)
    if raw is None:return dict(ml=np.nan,reason=reason,table=None)
    t=gate_table(raw,cc_min,snr_min);u=t[t.used]
    if len(u)<MIN_CHAN:return dict(ml=np.nan,reason=f'{len(u)} usable channels (<{MIN_CHAN})',table=t,n_used=len(u))
    dml=float(u.dML.median());sd=float((u.dML.quantile(.75)-u.dML.quantile(.25))/1.349) if len(u)>2 else np.nan
    return dict(ml=ml_tpl+dml,dML=dml,ml_template=ml_tpl,n_channels=len(u),n_stations=u.station.nunique(),channel_sigma=sd,table=t,reason='')

def event_magnitude(sh,cusp,tpl,origin,hits,cc_min=CC_MIN,snr_min=SNR_MIN,max_templates=None,present=None):
    """ML of one detection from EVERY template that detected it (hits = template cuspids, best-CC first)."""
    per=[]
    for c in (hits[:max_templates] if max_templates else hits):
        if int(c) not in tpl.index or int(c) not in cusp:continue
        row=tpl.loc[int(c)];ti=cusp.index(int(c))
        e=template_estimate(sh,ti,origin,float(row.ML) if pd.notna(row.ML) else np.nan,row.origin_time_UTC,cc_min,snr_min,present)
        if np.isfinite(e.get('ml',np.nan)):per.append(dict(template_cuspid=int(c),ml=e['ml'],dML=e['dML'],ml_template=e['ml_template'],
                                                           n_channels=e['n_channels'],n_stations=e['n_stations'],channel_sigma=e['channel_sigma']))
    if not per:return dict(ml=np.nan,reason='no template gave an estimate',n_templates=0,per_template=pd.DataFrame())
    p=pd.DataFrame(per);ml=float(p.ml.median());n=len(p)
    tsc=float((p.ml.quantile(.75)-p.ml.quantile(.25))/1.349) if n>2 else (float(abs(p.ml.diff().iloc[-1])/1.414) if n==2 else np.nan)
    cal,held=_cal();nst=int(p.n_stations.max());sup='1 link' if nst==1 else ('2-3 links' if nst<=3 else ('4-9 links' if nst<=9 else '>=10 links'))
    rms=float(held.loc[sup,'rms']) if sup in held.index else float(held.loc['all','rms'])
    sem=(tsc/np.sqrt(n)) if (np.isfinite(tsc) and n>1) else np.nan
    sig=float(np.sqrt(rms**2+(sem**2 if np.isfinite(sem) else 0.0)))
    return dict(ml=ml,sigma=sig,rms_heldout=rms,template_scatter=tsc,n_templates=n,n_stations=nst,n_channels=int(p.n_channels.sum()),
                support=sup,per_template=p,reason='')

def _load():
    sh=S.load_shelly();tpl=pd.read_csv(TPL).set_index('cuspid');return sh,list(sh['cuspid'].astype(int)),tpl

def _hits_by_event(cat,raw):
    """For each declustered event, the cuspids of every raw hit within +-MATCH_S, best CC first. Hits of the excluded templates
    (build_catalog.EXCLUDED_TEMPLATES; decision 20, 2026-10-06: dropped as if they had never been templates) are not used."""
    from build_catalog import EXCLUDED_TEMPLATES
    raw=raw[~raw.template_cuspid.astype(int).isin(list(EXCLUDED_TEMPLATES))]
    r=raw.sort_values('cc',ascending=False);rt=pd.to_datetime(r.origin).values.astype('datetime64[ns]').astype('int64')/1e9
    out={}
    for o in cat.origin:
        t=pd.Timestamp(o).value/1e9;m=np.abs(rt-t)<=MATCH_S
        out[o]=list(dict.fromkeys(r.template_cuspid.values[m].astype(int)))
    return out

if __name__=='__main__':
    mode=sys.argv[1];sh,cusp,tpl=_load()
    if mode in ('--catalog','--sweep','--selftest'):
        cat=pd.read_csv(C.CATALOG_CSV);raw=pd.read_csv(C.CATALOG_CSV.replace('.csv','_raw.csv'))
        if mode=='--selftest':cat=cat[cat.self_detection].head(120)
        elif mode=='--sweep':cat=cat[cat.self_detection].head(60)
        hits=_hits_by_event(cat,raw)
        if mode=='--sweep':
            # measure every (event, template) link ONCE, then apply each gate combination to the stored tables
            meas=[];print(f'measuring {len(cat)} events...',flush=True)
            for k,r in enumerate(cat.itertuples()):
                ml_tpl_self=tpl.ML.get(int(r.template_cuspid),np.nan);per=[]
                for c in hits[r.origin]:
                    if int(c) not in tpl.index or int(c) not in cusp:continue
                    row=tpl.loc[int(c)];t,_=measure(sh,cusp.index(int(c)),r.origin,float(row.ML) if pd.notna(row.ML) else np.nan,row.origin_time_UTC)
                    if t is not None and len(t):per.append((float(row.ML),t))
                if per:meas.append((ml_tpl_self,per))
                if (k+1)%20==0:print(f'  {k+1}/{len(cat)}',flush=True)
            print(f'measured {len(meas)} events; sweeping gates (truth: ML == the self-template ML)',flush=True)
            for cc in (0.3,0.4,0.5,0.6,0.7,0.8):
                for sn in (0.0,1.5,2.0,3.0):
                    err=[];nt=[]
                    for ml_true,per in meas:
                        mls=[]
                        for ml_t,t in per:
                            u=gate_table(t,cc,sn);u=u[u.used]
                            if len(u)>=MIN_CHAN:mls.append(ml_t+float(u.dML.median()))
                        if mls and np.isfinite(ml_true):err.append(float(np.median(mls))-ml_true);nt.append(len(mls))
                    if err:print(f'  CC>={cc} SNR>={sn}: n={len(err):3d}  bias={np.median(err):+.3f}  rms={np.std(err):.3f}  MAE={np.mean(np.abs(err)):.3f}  median templates={np.median(nt):.0f}',flush=True)
            sys.exit()
        rows=[]
        for r in cat.itertuples():
            e=event_magnitude(sh,cusp,tpl,r.origin,hits[r.origin],max_templates=MAX_TEMPLATES)
            rows.append(dict(origin=r.origin,template_cuspid=int(r.template_cuspid),cc=r.cc,ml=e.get('ml'),sigma=e.get('sigma'),
                             template_scatter=e.get('template_scatter'),n_templates_used=e.get('n_templates',0),n_stations=e.get('n_stations'),
                             n_channels=e.get('n_channels'),support=e.get('support'),reason=e.get('reason')))
            if len(rows)%100==0:print(f'  {len(rows)}/{len(cat)}',flush=True)
        d=pd.DataFrame(rows);suf='' if (C.FMF_RUN in ('','full_period')) else f'_{C.FMF_RUN}'
        f=os.path.join(os.path.dirname(os.path.abspath(__file__)),'logs',f'detection_magnitudes{suf}.csv');d.to_csv(f,index=False)
        print('wrote',len(d),'->',f)
        if mode=='--selftest':
            d['ml_tpl']=tpl.ML.reindex(d.template_cuspid).values;v=d[d.ml.notna()&d.ml_tpl.notna()]
            print(f'SELF-TEST n={len(v)}: median(ML-ML_tpl)={float((v.ml-v.ml_tpl).median()):+.4f}  rms={float((v.ml-v.ml_tpl).std()):.4f}  max|d|={float((v.ml-v.ml_tpl).abs().max()):.4f}')
    else:
        origin=sys.argv[1];cat=pd.read_csv(C.CATALOG_CSV);raw=pd.read_csv(C.CATALOG_CSV.replace('.csv','_raw.csv'))
        hits=[int(x) for x in sys.argv[2:]] or _hits_by_event(cat[cat.origin.str.startswith(origin[:19])],raw).get(cat[cat.origin.str.startswith(origin[:19])].origin.iloc[0],[])
        e=event_magnitude(sh,cusp,tpl,origin,hits)
        print(f"ML = {e['ml']:.2f} +- {e['sigma']:.2f}   ({e['n_templates']} templates, {e['n_stations']} stations, {e['n_channels']} channel measurements, "
              f"held-out rms {e['rms_heldout']:.2f}, inter-template scatter {e['template_scatter']:.3f})" if np.isfinite(e.get('ml',np.nan)) else f"no estimate: {e['reason']}")
        if len(e['per_template']):print(e['per_template'].round(3).to_string(index=False))
