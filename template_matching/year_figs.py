"""One completed year of the refined 5-20 Hz run, every catalogue detection shown (nb48 input; generalises burst_figs.py of nb45).
usage (ONLY FMF_RUN exported; config.py loads the run's settings): cd fmf && FMF_RUN=full_period_5_20H /home/msseo/miniforge3/envs/fmf/bin/python year_figs.py YEAR
Classes follow the final catalogue (decision 20): a detection is a reference event (is_reference) or new (is_new), nothing else.
self_detection (the detecting template's own event) is carried as information only (nb48-nb51 use it to take the template's ML).
Detections are grouped by the template that made them (one group = one source family); each group gets a repeater gather at the station
with the highest S correlation (template on top, every detection of that template in the year beneath, aligned on the template's P pick),
and every detection gets the nb40/nb45 record section (template windows in place on the continuous record + the detector's CC trace).
Relative size against the detecting template, as burst_figs.py: per S channel, the least-squares amplitude ratio of the detection's S window
on the template's at the lag (|lag| <= 0.3 s) of best correlation; channels kept with CC >= CC_MIN and SNR >= SNR_MIN on both windows (noise =
origin-6 to origin-1 s); dM = median log10 ratio (slope 1, as nb43). Outputs: notebooks/figs_pub/joint_fmf_<YEAR>/."""
import os, sys, shutil, subprocess

def _link(src, dst):   # decision 25 (one home): dst becomes a relative link to the product src, never a byte copy
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.relpath(src, os.path.dirname(os.path.abspath(dst))), dst)
import numpy as np, pandas as pd
from obspy import UTCDateTime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C, similarity as S
from station_gather import station_gather
YEAR = int(sys.argv[1])
HERE = os.path.dirname(os.path.abspath(__file__))
import pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # registry paths
OUT = os.path.join(haenam.path(None, 'dl_notebooks'), 'figs_pub', f'joint_fmf_{YEAR}'); os.makedirs(OUT, exist_ok=True)
CC_MIN = 0.6; SNR_MIN = 2.0; LAG = 30
sh = S.load_shelly(); cusp = list(sh['cuspid'].astype(int)); stations = [str(x) for x in sh['stations']]; prefixes = [str(p) for p in sh['channels']]; comps = [str(c) for c in sh['components']]
if not C.FMF_RUN: raise SystemExit('run with FMF_RUN set (e.g. FMF_RUN=full_period_5_20H) and no other FMF_* variable')
cat = pd.read_csv(C.CATALOG_CSV, parse_dates=['origin'], dtype={'reference_event_id': str})
if not {'is_reference', 'is_new'} <= set(cat.columns) or (cat.is_reference == cat.is_new).any():
    raise SystemExit(f'{C.CATALOG_CSV} is not a final two-class catalogue (is_reference / is_new)')
S._use_helvetica()                                                   # the gathers are drawn in this process

def rms(x): return float(np.sqrt(np.mean(np.square(x - x.mean())))) if x is not None and x.size else np.nan
def relsize(ti, origin):
    o = UTCDateTime(pd.Timestamp(origin).to_pydatetime()); to = UTCDateTime(str(sh['origin_utc'][ti])); rows = []
    for si, ph, ci, mo, n, tw, j in S.channels_of(sh, ti):
        if ph != 'S': continue
        net, sta = stations[si].split('.'); st = o + float(sh['ref_s'][ti]) + mo / C.FS
        d = S.read_window(net, sta, comps[ci], prefixes[si], st - LAG / C.FS, n + 2 * LAG)
        nz = S.read_window(net, sta, comps[ci], prefixes[si], o - 6.0, int(5 * C.FS)); nt = S.read_window(net, sta, comps[ci], prefixes[si], to - 6.0, int(5 * C.FS))
        if d is None or nz is None or nt is None: continue
        t = tw - tw.mean(); best = (0.0, LAG, np.nan)
        for L in range(2 * LAG + 1):
            x = d[L:L + n]; x = x - x.mean(); den = np.sqrt((x * x).sum() * (t * t).sum())
            if den > 0:
                cc = float((x * t).sum() / den)
                if abs(cc) > abs(best[0]): best = (cc, L, float((x * t).sum() / (t * t).sum()))
        cc, L, r = best; x = d[L:L + n]
        rows.append(dict(station=stations[si], comp=comps[ci], cc=cc, lag_s=(L - LAG) / C.FS, ratio=r, snr_det=rms(x) / rms(nz), snr_tpl=rms(tw) / rms(nt)))
    return pd.DataFrame(rows)

det = cat[cat.origin.dt.year == YEAR].sort_values('origin').reset_index(drop=True)
rows, chans, sets = [], [], []
for r in det.itertuples():
    c = int(r.template_cuspid); ti = cusp.index(c); tref = pd.Timestamp(str(sh['origin_utc'][ti])[:23])
    m = relsize(ti, r.origin); g = m[(m.cc >= CC_MIN) & (m.snr_det >= SNR_MIN) & (m.snr_tpl >= SNR_MIN) & (m.ratio > 0)]
    dM = float(np.median(np.log10(g.ratio))) if len(g) >= 2 else np.nan
    best = m.loc[m.cc.abs().idxmax()] if len(m) else None
    rows.append(dict(origin=r.origin, template=c, template_origin=tref, days_from_template=(r.origin - tref).total_seconds() / 86400, network_cc=float(r.cc),
                     n_templates=int(r.n_templates), n_present=int(r.n_present), is_reference=bool(r.is_reference), is_new=bool(r.is_new),
                     reference_event_id=r.reference_event_id if isinstance(r.reference_event_id, str) else '', reference_offset_s=float(r.reference_offset_s), self_detection=bool(r.self_detection),
                     best_S_station=best.station if best is not None else '', best_S_cc=float(best.cc) if best is not None else np.nan, n_ratio_channels=len(g), dM=dM))
    for q in m.itertuples():
        chans.append(dict(origin=r.origin, template=c, station=q.station, comp=q.comp, cc=q.cc, lag_s=q.lag_s, ratio=q.ratio, snr_det=q.snr_det, snr_tpl=q.snr_tpl,
                          used=bool(q.cc >= CC_MIN and q.snr_det >= SNR_MIN and q.snr_tpl >= SNR_MIN and q.ratio > 0)))
    # record section of every detection with the template that made it (the template's own event included)
    k = pd.Timestamp(r.origin).strftime('%Y-%m-%d %H:%M:%S.%f')[:23]
    subprocess.run([sys.executable, os.path.join(HERE, 'band_event_figs.py'), k, str(c)], check=True, capture_output=True, env=dict(os.environ, FMF_SET_ONLY='1'))
    stem = f"{pd.Timestamp(k):%Y%m%d_%H%M%S}_{c}"; png = f"S_{stem}.png"
    _link(os.path.join(HERE, f'figs_similarity_{C.FMF_RUN}', 'new', stem + '_set.png'), os.path.join(OUT, png))
    sets.append(dict(origin=r.origin, template=c, png=png)); print('record section', k, c, flush=True)
t = pd.DataFrame(rows)
# repeater gather per template, at the station with the highest S correlation among that template's detections in the year
gathers = []
for c, g in t.groupby('template', sort=False):
    sta = g.loc[g.best_S_cc.abs().idxmax(), 'best_S_station'] if g.best_S_station.ne('').any() else ''
    if not sta: continue
    png = f'G_{c}.png'
    station_gather(c, sta, f'{YEAR}-01-01', f'{YEAR + 1}-01-01', os.path.join(OUT, png), title=f'Template {c}: every {YEAR} detection at {sta}, aligned on the template’s P pick', same_template=True)
    gathers.append(dict(template=c, station=sta, png=png)); print('gather', c, sta, flush=True)
# day statistics of the detecting template (logged by band_event_figs): network CC as a multiple of that day's MAD
lg = pd.read_csv(os.path.join(HERE, 'logs', 'band_event_channels.csv'), parse_dates=['origin']); lg = lg[lg.run == C.FMF_RUN].drop_duplicates(['origin', 'template_cuspid'], keep='last')
for r in t.itertuples():
    q = lg[((lg.origin - r.origin).abs() < pd.Timedelta(seconds=0.5)) & (lg.template_cuspid == r.template)]
    if len(q): t.loc[r.Index, 'day_mad'] = float(q.day_mad.iloc[0]); t.loc[r.Index, 'day_threshold'] = float(q.day_thr.iloc[0])
t['cc_over_mad'] = t.network_cc / t.day_mad
t.to_csv(os.path.join(OUT, 'year_detections.csv'), index=False); pd.DataFrame(chans).to_csv(os.path.join(OUT, 'year_ratio_channels.csv'), index=False)
pd.DataFrame(sets).to_csv(os.path.join(OUT, 'year_set_figures.csv'), index=False); pd.DataFrame(gathers).to_csv(os.path.join(OUT, 'year_gathers.csv'), index=False)
print(t[['origin', 'template', 'network_cc', 'cc_over_mad', 'n_templates', 'best_S_station', 'best_S_cc', 'n_ratio_channels', 'dM']].round(3).to_string(index=False))
