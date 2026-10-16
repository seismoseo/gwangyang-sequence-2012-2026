"""Probe the first relocated 2020 event at JDO2/HAWA/WAN2: is the reported Wood-Anderson peak signal or background?

Diagnostic only; nothing here changes the magnitude recipe. Writes first_event_probe/ (figure, metrics.csv, arrays.npz).
Renewal R6 (2026-10-14): copy of docs/history/magnitudes/probe_first_event.py with registry paths: the R4 waveforms and the ML run
on the merged responses (JDO2 with NECIS's gain, ISSUES #40). Run: python magnitudes/local_ml/full_period/probe_first_event.py
"""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,obspy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
import pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam
sys.path.insert(0,str(haenam.ROOT/'magnitudes/local_ml'))
import haenam_magnitudes as H
sys.path.insert(0,str(haenam.path(None,'dl_notebooks')))
import joint_science as J
ML=haenam.ROOT/'magnitudes/local_ml/full_period';WF=Path(haenam.path(None,'reloc_final_run'))/'waveforms_100km'
OUT=J.setup(ML/'first_event_probe');eid='20200425131318'   # first relocated 2020 event; previous round JDO2-only ML 2.82
inv=obspy.read_inventory(str(ML/'responses_full_period.xml'))
co=pd.read_csv(ML/'component_measurements.csv',dtype={'eid':str},low_memory=False)
rows=[];arrays={};fig,axs=plt.subplots(3,2,figsize=(12,8),layout='constrained')
for row,sta in enumerate(['JDO2','HAWA','WAN2']):
    r=co[co.eid.eq(eid)&co.station.eq(sta)&co.component.eq('Z')].iloc[0]
    tr=obspy.read(str(WF/eid/r.file))[0]
    wa=H.S._wa_trace(tr,inv);tt=wa.times()+float(tr.stats.sac.b)
    lo,hi=r.signal_start_s,r.signal_end_s;sig=(tt>=lo)&(tt<hi);noise=(tt>=r.noise_start_s)&(tt<r.noise_end_s);pre=(tt>=-20)&(tt<r.noise_end_s)
    peak_t=tt[np.flatnonzero(sig)[np.argmax(abs(wa.data[sig]))]]
    f,psd=welch(wa.data[sig],fs=100,nperseg=min(512,sig.sum()))
    record=dict(station=sta,peak_t=peak_t,P=r.p_origin_s,S=float(tr.stats.sac.t0),signal_start_s=lo,signal_end_s=hi,noise_start_s=r.noise_start_s,noise_end_s=r.noise_end_s,signal_peak=float(abs(wa.data[sig]).max()),noise_peak=float(abs(wa.data[noise]).max()),pre20_peak=float(abs(wa.data[pre]).max()),signal_RMS=float(np.sqrt(np.mean(wa.data[sig]**2))),noise_RMS=float(np.sqrt(np.mean(wa.data[noise]**2))),dominant_Hz=float(f[np.argmax(psd)]))
    arrays[sta+'_t']=tt;arrays[sta+'_WA']=wa.data
    # Diagnostics only: assess whether deconvolution reintroduces out-of-band energy.
    for label,band in [('post_0.5_10',(0.5,10)),('post_2_10',(2,10)),('post_5_10',(5,10))]:
        w=wa.copy().filter('bandpass',freqmin=band[0],freqmax=band[1],corners=6,zerophase=True)
        record[label+'_peak']=float(abs(w.data[sig]).max());record[label+'_noise']=float(abs(w.data[noise]).max());arrays[sta+'_'+label]=w.data
    ax=axs[row,0];ax.plot(tt,wa.data,lw=.6);ax.axvspan(lo,hi,alpha=.12,color='orange');ax.axvspan(r.noise_start_s,r.noise_end_s,alpha=.12,color='gray');ax.scatter(peak_t,wa.data[np.argmin(abs(tt-peak_t))],c='red');ax.axvline(r.p_origin_s,c='k',ls=':',lw=.7);ax.axvline(float(tr.stats.sac.t0),c='k',ls='--',lw=.7);ax.set(xlim=(-20,25),ylabel='WA (mm)',title=sta+' inherited recipe')
    ax=axs[row,1]
    for label in ['WA','post_0.5_10','post_2_10','post_5_10']:ax.plot(tt,arrays[sta+'_'+label],lw=.7,label=label)
    ax.set(xlim=(2,17),ylabel='WA (mm)',title=sta+' diagnostic post-filters');ax.legend(fontsize=6)
    rows.append(record)
fig.savefig(OUT/'probe.png',dpi=180);plt.close(fig)
pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);np.savez(OUT/'arrays.npz',**arrays)
print(pd.DataFrame(rows).to_string(index=False),flush=True)
