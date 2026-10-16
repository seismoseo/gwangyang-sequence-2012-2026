"""Empirical station/era ML calibration and whole-event held-out validation.

Reads immutable Buan-recipe component amplitudes. Never modifies raw amplitudes,
responses, relocation, or the older magnitude products.
"""
import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # the project registry, found upward from this file
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
sys.path.insert(0,str(haenam.ROOT / 'magnitudes/local_ml'))
import buan_sheen2018_snapshot as S
MIN_CAL=10
MAX_SCATTER=.30
MAX_Q90=.50
MAX_ERA_SHIFT=.20
SNR_PRIMARY=3.

def load():
    folder=haenam.ROOT / 'magnitudes/local_ml/full_period'
    co=pd.read_csv(folder/'component_measurements.csv',dtype={'eid':str},low_memory=False)
    old=pd.read_csv(folder/'full_period_ML.csv',dtype={'eid':str,'event_id':str},parse_dates=['origin_time'])
    ref=pd.read_csv(haenam.ROOT / 'catalogs/reference_25m_607ev/DL_focused_25m.csv',dtype={'event_id':str,'template_match':str})
    old['era']=np.select([old.origin_time.lt('2020-08-01'),old.origin_time.lt('2026-08-14')],['2020','quiet'],default='2026')
    anchors=ref[ref.known_KMA_ML.notna()&~ref.template_ambiguous][['event_id','known_KMA_ML']].rename(columns={'event_id':'eid','known_KMA_ML':'anchor_ML'})
    assert anchors.eid.is_unique and len(anchors)==193
    # Reproduce the original amplitudes, but require an observed P for revised ML.
    co=co[co.status.eq('processed')&co.response_units_validated.eq(True)&co.p_source.eq('pick')].copy()
    assert not co.duplicated(['eid','network','station','component']).any()
    amp=co.pivot(index=['eid','network','station'],columns='component',values='peak_mm')
    snr=co.pivot(index=['eid','network','station'],columns='component',values='snr_corrected')
    dist=co.groupby(['eid','network','station']).dist_km.first()
    rows=[]
    for mode,values,scores in [('H',np.sqrt(amp.N*amp.E),np.minimum(snr.N,snr.E)),('V',amp.Z,snr.Z)]:
        tab=pd.DataFrame({'amplitude_mm':values,'snr':scores,'dist_km':dist}).reset_index()
        tab=tab[np.isfinite(tab.amplitude_mm)&tab.amplitude_mm.gt(0)&tab.snr.ge(2)].copy()
        coef=S.COEF[mode];R=tab.dist_km
        tab['station_ML_raw']=np.log10(tab.amplitude_mm)+coef['a']*np.log10(R/100)+coef['b']*(R-100)+3
        tab['mode']=mode;rows.append(tab)
    samples=pd.concat(rows,ignore_index=True).merge(old[['eid','origin_time','era']],on='eid',validate='many_to_one').merge(anchors,on='eid',how='left',validate='many_to_one')
    samples['residual']=samples.station_ML_raw-samples.anchor_ML
    return old,ref,samples

def summarize(g):
    r=g.residual.to_numpy();n=len(r)
    if not n:return dict(n=0,accepted=False)
    center=float(np.median(r));spread=float(1.4826*np.median(abs(r-center)))
    loo=np.array([r[i]-np.median(np.delete(r,i)) for i in range(n)]) if n>1 else np.array([np.nan])
    q90=float(np.quantile(abs(loo),.9)) if n>1 else np.inf
    return dict(n=n,correction=-center,robust_scatter=spread,LOO_abs_error_q90=q90,accepted=bool(n>=MIN_CAL and spread<=MAX_SCATTER and q90<=MAX_Q90),first_anchor=str(g.origin_time.min()),last_anchor=str(g.origin_time.max()))

def models(samples,exclude=None):
    train=samples[samples.anchor_ML.notna()&samples.snr.ge(SNR_PRIMARY)&samples.era.ne('quiet')]
    if exclude is not None:train=train[train.eid.ne(exclude)]
    rows=[];mapping={}
    for key,g in train.groupby(['network','station','mode'],sort=True):
        groups={era:g[g.era.eq(era)] for era in ['2020','2026']}
        stats={era:summarize(x) for era,x in groups.items()}
        stable=all(stats[e]['n']>=MIN_CAL for e in groups)
        shift=stats['2026']['correction']-stats['2020']['correction'] if stable else np.nan
        pooled=summarize(g)
        stable=bool(stable and abs(shift)<=MAX_ERA_SHIFT and pooled['accepted'])
        for era,st in stats.items():
            entry=dict(network=key[0],station=key[1],mode=key[2],scope=era,**st,stable_across_active_eras=stable,correction_2026_minus_2020=shift)
            rows.append(entry)
            if st['accepted']:mapping[key+(era,)]=entry
        pooledrow=dict(network=key[0],station=key[1],mode=key[2],scope='pooled',**pooled,stable_across_active_eras=stable,correction_2026_minus_2020=shift)
        pooledrow['accepted']=stable;rows.append(pooledrow)
        # Pooled corrections are used only where both endpoints independently support stability.
        if stable:
            for era in ['2020','quiet','2026']:mapping[key+(era,)]=pooledrow
    return pd.DataFrame(rows),mapping

def apply(samples,mapping,snr_min=SNR_PRIMARY):
    records=[]
    for r in samples[samples.snr.ge(snr_min)].itertuples():
        m=mapping.get((r.network,r.station,r.mode,r.era))
        if m is None:continue
        records.append(dict(eid=r.eid,network=r.network,station=r.station,mode=r.mode,snr=r.snr,station_ML_raw=r.station_ML_raw,correction=m['correction'],station_ML_corrected=r.station_ML_raw+m['correction'],calibration_scope=m['scope'],n_calibration=m['n'],calibration_scatter=m['robust_scatter'],calibration_q90=m['LOO_abs_error_q90']))
    cols=['eid','network','station','mode','snr','station_ML_raw','correction','station_ML_corrected','calibration_scope','n_calibration','calibration_scatter','calibration_q90']
    out=pd.DataFrame(records,columns=cols)
    # Horizontal first, otherwise a separately calibrated vertical estimate; no double counting.
    return out.sort_values('mode').drop_duplicates(['eid','network','station']).reset_index(drop=True)

def aggregate(st):
    rows=[]
    for eid,g in st.groupby('eid',sort=False):
        vals=g.station_ML_corrected.to_numpy();row=dict(eid=eid,n_corrected_stations=len(g),corrected_station_scatter=float(np.std(vals,ddof=1)) if len(vals)>1 else np.nan)
        for n in [1,2,3]:row[f'ml_corrected_min{n}']=float(np.median(vals)) if len(vals)>=n else np.nan
        rows.append(row)
    return pd.DataFrame(rows,columns=['eid','n_corrected_stations','corrected_station_scatter','ml_corrected_min1','ml_corrected_min2','ml_corrected_min3'])

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);old,ref,samples=load()
    table,mapping=models(samples);station=apply(samples,mapping);agg=aggregate(station)
    new=old.merge(agg,on='eid',how='left',validate='one_to_one');new.n_corrected_stations=new.n_corrected_stations.fillna(0).astype(int)
    for n in [1,2,3]:assert new[f'ml_corrected_min{n}'].notna().equals(new.n_corrected_stations.ge(n))
    # This is a reporting comparison, never the default revised magnitude.
    sensitivity=aggregate(apply(samples,mapping,snr_min=2)).add_suffix('_SNR2').rename(columns={'eid_SNR2':'eid'})
    new=new.merge(sensitivity,on='eid',how='left',validate='one_to_one')
    # Every held-out event is removed from every station, mode and era-calibration decision.
    cv=[];anchors=samples[samples.anchor_ML.notna()].groupby('eid').anchor_ML.first()
    for i,(eid,ml) in enumerate(anchors.items()):
        _,fold=models(samples,exclude=eid);st=apply(samples[samples.eid.eq(eid)],fold);ag=aggregate(st)
        cv.append(dict(eid=eid,KMA_ML=ml,n_stations=len(st),ML_corrected_CV=float(ag.ml_corrected_min3.iloc[0]) if len(ag) else np.nan,ML_one_station_CV=float(ag.ml_corrected_min1.iloc[0]) if len(ag) else np.nan))
        if (i+1)%40==0:print('Held-out validation',i+1,'/',len(anchors),flush=True)
    cv=pd.DataFrame(cv).merge(old[['eid','ml_min3','era']],on='eid');cv['error_corrected']=cv.ML_corrected_CV-cv.KMA_ML;cv['error_legacy']=cv.ml_min3-cv.KMA_ML
    # Report only actually measured coverage; changes do not imply changed detection completeness.
    coverage=[]
    for catalog,g in [('All relocated',new),('25 m focused reference',new[new.eid.isin(ref.event_id)])]:
        for era,x in [('Full period',g),*[(e,g[g.era.eq(e)]) for e in ['2020','quiet','2026']]]:
            coverage.append(dict(catalog=catalog,era=era,N=len(x),legacy_ML3=int(x.ml_min3.notna().sum()),legacy_ML1=int(x.ml_min1.notna().sum()),corrected_ML3=int(x.ml_corrected_min3.notna().sum()),corrected_ML2=int(x.ml_corrected_min2.notna().sum()),corrected_ML1=int(x.ml_corrected_min1.notna().sum()),corrected_ML1_SNR2=int(x.ml_corrected_min1_SNR2.notna().sum())))
    good=cv[cv.ML_corrected_CV.notna()];same=good[good.ml_min3.notna()]
    validation=dict(n_primary_heldout=len(good),median_error=float(good.error_corrected.median()),RMSE=float(np.sqrt(np.mean(good.error_corrected**2))),MAE=float(good.error_corrected.abs().mean()),q90_abs_error=float(good.error_corrected.abs().quantile(.9)),same_event_comparison_n=len(same),same_event_legacy_RMSE=float(np.sqrt(np.mean(same.error_legacy**2))),same_event_corrected_RMSE=float(np.sqrt(np.mean(same.error_corrected**2))),excluded_first_event_has_no_reported_ML=bool(new.set_index('eid').loc['20200425131318','ml_corrected_min1']!=new.set_index('eid').loc['20200425131318','ml_corrected_min1']))
    assert validation['excluded_first_event_has_no_reported_ML']
    for name,df in [('full_period_station_corrected_ML',new),('corrected_station_measurements',station),('station_corrections',table),('calibration_amplitude_candidates',samples),('heldout_event_validation',cv),('coverage_comparison',pd.DataFrame(coverage))]:df.to_csv(out/(name+'.csv'),index=False)
    paths=[Path(__file__),Path(S.__file__),haenam.ROOT / 'magnitudes/local_ml/full_period/component_measurements.csv',haenam.ROOT / 'magnitudes/local_ml/full_period/full_period_ML.csv',haenam.ROOT / 'catalogs/reference_25m_607ev/DL_focused_25m.csv']
    metadata=dict(method='Empirical KMA-anchored additive station/era H/V corrections; original waveform amplitudes retained',default_peak_SNR=SNR_PRIMARY,minimum_station_calibration_events=MIN_CAL,maximum_robust_station_scatter=MAX_SCATTER,maximum_station_LOO_q90=MAX_Q90,maximum_era_correction_difference_for_pooling=MAX_ERA_SHIFT,epochs={'2020':'UTC before 2020-08-01','quiet':'2020-08-01 to 2026-08-14 exclusive','2026':'UTC on/after 2026-08-14'},quiescence='Only pooled corrections supported in both active eras; unobserved intermediate drift remains a limitation',anchors='Unambiguous known KMA magnitudes in frozen focused reference; quiescence anchor excluded from calibration',gates='Observed P and peak SNR>=3 per used component; both E/N required for H; H preferred, separately calibrated V fallback; at least 3 stations for primary',validation=validation,inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
    (out/'provenance.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(validation,indent=2),flush=True);print(pd.DataFrame(coverage).to_string(index=False),flush=True)
    return new,table,cv

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();run(args.out)
