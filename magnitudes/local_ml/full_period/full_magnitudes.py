"""Full-period Buan-method ML; reads original SAC files in place.

Preserves the previous Haenam KS/KG engine plus full-SEED-ID JU response handling,
SAC epicentral distances, new joint-run depth, corrected theoretical-P timing,
SNR>=2, horizontal preference, vertical fallback, and >=3 picked stations.
"""
import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # the project registry, found upward from this file
from pathlib import Path
import sys,json,hashlib,time,os,traceback
import numpy as np
import pandas as pd
import obspy
sys.path.insert(0,str(haenam.ROOT / 'magnitudes/local_ml'))
import haenam_magnitudes as H
import jn_magnitudes as J
import haenam   # registry (on sys.path through haenam_magnitudes)
S=H.S
WF=Path(haenam.path(None,'reloc_final_run'))/'waveforms_100km'   # the final relocation run's exports (as adopt_ext0922_reference.sh ran it)
OUT=None
# Renewal R6 (2026-10-14): KS/KG responses from the merged inventory (ISSUES #40) in place of the Buan master/fetched set;
# the window spans the R4 events (2018-2026); NAHA left out (its listed site is wrong, ISSUES #39). KWJU and SEJF carry no
# channel in the merged inventory, so they drop out as stations without a response.
RESP_XML=haenam.ROOT / 'docs/response_inventory/merged_inventory.xml'   # GWANGYANG PORT: the Gwangyang merged inventory (G6)
WINDOW=('2012-12-01','2026-10-07')   # GWANGYANG PORT: the study window
EXCLUDE_STATIONS=[]   # GWANGYANG PORT: NAHA kept; its corrected site (34.8167, 127.9261) is in the station table, so SAC distances are right (G-D17)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def code_key(path=__file__):
    """sha256 of this module's code with every path taken out: the code part of the cache key.
    Path strings (any string with a '/'), Path(...) and haenam.path(...) calls, '/' joins with a path and names assigned
    from a path become one placeholder, and module-level path assignments are dropped; comments and layout never count.
    Moving the project, which edits only paths, therefore keeps the cache; any other code edit invalidates it."""
    import ast
    tree=ast.parse(Path(path).read_text());names=set()
    def is_path(n):
        if isinstance(n,ast.Constant):return isinstance(n.value,str) and '/' in n.value
        if isinstance(n,ast.JoinedStr):return any(is_path(v) for v in n.values)
        if isinstance(n,ast.Name):return n.id in names
        if isinstance(n,ast.Attribute):return is_path(n.value) or (isinstance(n.value,ast.Name) and n.value.id=='haenam' and n.attr=='ROOT')
        if isinstance(n,ast.Call):
            f=n.func
            return (isinstance(f,ast.Name) and f.id=='Path') or (isinstance(f,ast.Attribute) and (f.attr=='path' and isinstance(f.value,ast.Name) and f.value.id=='haenam' or f.attr in ('resolve','absolute','expanduser') and is_path(f.value)))
        if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Div):return is_path(n.left) or is_path(n.right)
        return False
    body=[]
    for st in tree.body:          # module level: NAME = <path> is configuration, not code
        if isinstance(st,ast.Assign) and len(st.targets)==1 and isinstance(st.targets[0],ast.Name) and is_path(st.value):
            names.add(st.targets[0].id);continue
        body.append(st)
    tree.body=body
    class Strip(ast.NodeTransformer):
        def visit(self,n):
            if isinstance(n,ast.expr) and is_path(n):return ast.Constant('<path>')
            return super().visit(n)
    tree=Strip().visit(tree)
    return hashlib.sha256(ast.dump(tree,annotate_fields=False,include_attributes=False).encode()).hexdigest()

def initialize(out):
    global OUT
    OUT=Path(out);OUT.mkdir(parents=True,exist_ok=True);(OUT/'event_cache').mkdir(exist_ok=True);(OUT/'combined_cache').mkdir(exist_ok=True)
    invpath=OUT/'responses_full_period.xml'
    if not invpath.exists():
        print('Loading the merged response inventory',flush=True)
        big=obspy.read_inventory(str(RESP_XML))
        from obspy.core.inventory import Inventory
        inv=Inventory(networks=[],source='Gwangyang merged inventory (docs/response_inventory), Gwangyang KS/KG stations')   # GWANGYANG PORT
        stations=pd.read_csv(Path(haenam.path(None, 'dl_station_table')))
        for r in stations[stations.Network.isin(['KS','KG'])&~stations.Code.isin(EXCLUDE_STATIONS)].itertuples():
            inv+=big.select(network=r.Network,station=r.Code,starttime=obspy.UTCDateTime(WINDOW[0]),endtime=obspy.UTCDateTime(WINDOW[1]))
        inv.write(str(invpath),format='STATIONXML')
        H.INV=inv
    else:H.INV=obspy.read_inventory(str(invpath))
    from obspy.taup import TauPyModel
    H.TAUP=TauPyModel(model=str(H.BUAN/'taup/kim1983_buan.npz'));S._KIM1983_TAUP=H.TAUP
    H.WF=WF;H.OUT=OUT
    # Cache invalidates on code (paths excluded, code_key), inventory or method changes as well as waveform stat signatures/depth.
    H.VERSION='full-ml-v2-'+sha(invpath)[:12]+'-'+code_key()[:12]
    J.INV=obspy.read_inventory(str(J.XML));J.TAUP=H.TAUP;J.JN=sorted({sta.code for net in J.INV for sta in net if sta.code.startswith('JN')})
    print('Response inventory ready; JU stations',J.JN,flush=True)


def event(row):
    p=OUT/'combined_cache'/(row['eid']+'.json')
    old=H.measure_one(row)
    version=H.VERSION+'-'+sha(J.XML)[:12]
    if p.exists():
        cached=json.loads(p.read_text())
        if cached.get('version')==version and cached.get('waveform_signature')==old['waveform_signature'] and cached.get('depth_km')==row['depth_km']:return cached
    # JN event uses the same H.WF/H helpers and full SEED ID response epochs.
    co,st,files=J.event(row)
    stations=[r for r in old['stations'] if r['mode']=='corrected']+st
    components=[r for r in old['components'] if r.get('status')!='network_excluded_buan']+co
    tab=pd.DataFrame(stations)
    if len(tab):assert not tab.duplicated(['network','station']).any()
    agg=S.event_ml(tab)
    if len(tab):
        mlsta=tab.ml_h.where(tab.ml_h.notna(),tab.ml_v);med=float(mlsta.median());spread=float(mlsta.std()) if len(mlsta)>1 else np.nan
        hp=tab.ml_h.notna()&tab.p_source.eq('pick')
        agg.update(ml_min1=med if agg['n_pick']>=1 else np.nan,ml_min2=med if agg['n_pick']>=2 else np.nan,ml_min3=agg['ml'],station_scatter=spread,ml_h_strict=float(tab.ml_h.median()) if hp.sum()>=3 else np.nan,n_JN=int(tab.network.eq('JU').sum()),n_JN_picked=int((tab.network.eq('JU')&tab.p_source.eq('pick')).sum()))
    else:agg.update(ml_min1=np.nan,ml_min2=np.nan,ml_min3=np.nan,station_scatter=np.nan,ml_h_strict=np.nan,n_JN=0,n_JN_picked=0)
    result=dict(version=version,eid=row['eid'],depth_km=row['depth_km'],waveform_signature=old['waveform_signature'],waveform_files=old['waveform_files'],event=agg,stations=stations,components=components,JN_sources=files,ml_KS_KG=old['event']['corrected']['ml'])
    p.write_text(json.dumps(result,allow_nan=True));return result


def run(out,workers=4,limit=None):
    initialize(out)
    cat=pd.read_csv(Path(haenam.path(None, 'dl_relocated_full')),dtype={'event_id':str,'template_match':str});cat['eid']=cat.event_id
    if limit:cat=cat.head(limit)
    from multiprocessing import get_context
    events=[];stations=[];components=[];jn=[];start=time.monotonic()
    with get_context('fork').Pool(workers) as pool:
        for i,r in enumerate(pool.imap_unordered(event,cat.to_dict('records')),1):
            events.append(dict(eid=r['eid'],**r['event'],ml_KS_KG=r['ml_KS_KG'],waveform_signature=r['waveform_signature'],waveform_files=r['waveform_files']))
            stations.extend(r['stations']);components.extend(r['components']);jn.extend(r['JN_sources'])
            if i%25==0 or i==len(cat):print(f'{i}/{len(cat)} events, {time.monotonic()-start:.0f}s; ML3 {sum(np.isfinite(x["ml_min3"]) for x in events)}, ML1 {sum(np.isfinite(x["ml_min1"]) for x in events)}',flush=True)
    ev=cat.merge(pd.DataFrame(events),on='eid',validate='one_to_one');co=pd.DataFrame(components);st=pd.DataFrame(stations)
    for name,df in [('full_period_ML',ev),('station_magnitudes',st),('component_measurements',co),('JN_waveform_hashes',pd.DataFrame(jn))]:df.to_csv(OUT/(name+'.csv'),index=False)
    for n in [1,2,3]:assert ev['ml_min'+str(n)].notna().equals(ev.n_pick>=n)
    co.groupby(['status','network']).size().to_csv(OUT/'component_status_counts.csv')
    sources=[Path(__file__),Path(H.__file__),Path(J.__file__),Path(S.__file__),J.XML,RESP_XML,OUT/'responses_full_period.xml',H.BUAN/'taup/kim1983_buan.npz',Path(haenam.path(None, 'dl_relocated_full'))]
    prov=dict(recipe='Prior Buan KS/KG processing plus prior Haenam JU full-SEED-ID extension; KS/KG responses from the Gwangyang merged inventory (NAHA kept, corrected site)',distance='SAC epicentral distance preserved to reproduce previous recipe; hypocentral timing uses joint depth',gates='SNR >=2; median H-preferred/V-fallback station ML; primary >=3 picked measured stations; 1/2 station results are sensitivity estimates',n_events=len(ev),n_ML3=int(ev.ml_min3.notna().sum()),n_ML1=int(ev.ml_min1.notna().sum()),waveforms='Read in place; per-event SHA256 of filename/size/mtime signatures; JU files additionally content-hashed',inputs=[dict(path=str(p),sha256=sha(p)) for p in sources])
    (OUT/'provenance.json').write_text(json.dumps(prov,indent=2));print(json.dumps({k:v for k,v in prov.items() if k!='inputs'},indent=2),flush=True)
    return ev

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int);a=p.parse_args();run(a.out,a.workers,a.limit)
