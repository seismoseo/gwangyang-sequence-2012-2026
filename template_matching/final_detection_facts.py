"""Per-detection facts of the FINAL catalog (run full_period_5_20H, decision 20: reference event or new) for nb37 and nb39.

The per-run tables of the FMF tools (logs/candidate_pick_support_<run>.csv from validate_candidate.py --all,
logs/similarity_channels_<run>.csv from channel_cc.py, logs/similarity_summary_<run>.csv from make_similarity_figs.py) were made
for the as-run catalogue. Most final detections have the same origin and representative template there; the others (where an
excluded template had represented the detection, or where the as-run rules had removed or set it aside) do not. This script
writes ONE table per fact set, keyed on the final catalog's (origin, template_cuspid), with fact columns only (no class or label
columns): values copied from the per-run tables where the key is there, computed here with the same functions otherwise
(validate_candidate.evidence; similarity.detection_vs_template / template_cc_day / cc_at, as channel_cc.py and
make_similarity_figs.py do). The per-run tables and figures are read only.

  logs/full_period_5_20H/final_catalog_facts/detection_picks.csv       every detection: PhaseNet picks at the template stations
  logs/full_period_5_20H/final_catalog_facts/detection_channels.csv    every detection that is not its template's own event:
                                                                       per-channel CC with the template (long format)
  logs/full_period_5_20H/final_catalog_facts/detection_similarity.csv  the same detections: network CC recomputed, channel counts,
                                                                       whole-day threshold, day-trace peak, prominence
  --gallery REQUEST.csv (origin, template_cuspid, label): per-channel overlay and day CC trace for each requested detection, titled
  with the final class, in figs_similarity_full_period_5_20H/final_gallery/<stamp>_<cuspid>_{overlay,cc}.png (nb39's gallery).

Run with ONLY FMF_RUN set:  cd template_matching && FMF_RUN=full_period_5_20H /home/msseo/miniforge3/envs/fmf/bin/python final_detection_facts.py [--gallery REQUEST.csv]
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if os.environ.get('FMF_RUN') != 'full_period_5_20H':
    raise SystemExit('run with FMF_RUN=full_period_5_20H (and nothing else set)')
import config as C
import similarity as S
import validate_candidate as V

RUN = C.FMF_RUN
FROZEN = os.path.join(HERE, 'catalogs', 'FROZEN_full_period_5_20H_final_20261006.json')
CAT = os.path.join(HERE, 'catalogs', 'fmf_haenam_catalog_full_period_5_20H_seg.csv')
OUT = os.path.join(HERE, 'logs', RUN, 'final_catalog_facts')
FIG = os.path.join(HERE, f'figs_similarity_{RUN}', 'final_gallery')
PICKS_RUN = os.path.join(HERE, 'logs', f'candidate_pick_support_{RUN}.csv')
CH_RUN = os.path.join(HERE, 'logs', f'similarity_channels_{RUN}.csv')
SUM_RUN = os.path.join(HERE, 'logs', f'similarity_summary_{RUN}.csv')
PICK_COLS = ['template_stations', 'picks_at_template_stations', 'picks', 'matched_phases', 'matched_stations', 'matched', 'matched_rms_s',
             's_minus_p', 'other_picks_in_window']
SIM_COLS = ['net_cc', 'mean_channel_cc', 'n_channels', 'n_channels_ge05', 'top1', 'top3_mean', 'wholeday_thr', 'day_mad', 'passes_wholeday',
            'peak_cc', 'peak_offset_s', 'second_peak_day', 'prominence', 'n_live_channels']


def catalog():
    fz = json.load(open(FROZEN))
    if hashlib.sha256(open(CAT, 'rb').read()).hexdigest() != fz['sha256']['catalogs/fmf_haenam_catalog_full_period_5_20H_seg.csv']:
        raise SystemExit('the final catalog differs from its FROZEN record')
    c = pd.read_csv(CAT, dtype={'origin': str, 'reference_event_id': str})
    c['reference_event_id'] = c.reference_event_id.fillna('').str.replace(r'\.0$', '', regex=True)
    c['key'] = c.origin + '|' + c.template_cuspid.astype(int).astype(str)
    return c


def keyed(df):
    df = df.copy(); df['key'] = df.origin.astype(str) + '|' + df.template_cuspid.astype(int).astype(str); return df


def similarity_row(sh, ti, origin, day):
    """make_similarity_figs.py's per-detection computation, unchanged."""
    y, doy = pd.Timestamp(origin).year, pd.Timestamp(origin).dayofyear
    ch, net_cc, mean_cc = S.detection_vs_template(sh, ti, origin, present=day[1])
    res = S.template_cc_day(sh, ti, y, doy, day=day)
    i, j, pk = S.cc_at(res, sh, ti, origin) if res is not None else (None, None, np.nan)
    if res is not None:
        cc = res['cc']; h = int(4 * C.FS / C.STEP); m = cc.copy(); m[max(0, i - h):i + h + 1] = -np.inf; second = float(m.max())
    else:
        second = np.nan
    top = sorted([x['cc'] for x in ch], reverse=True)
    row = dict(net_cc=net_cc, mean_channel_cc=mean_cc, n_channels=len(ch), n_channels_ge05=int(sum(v >= 0.5 for v in top)), top1=top[0] if top else np.nan,
               top3_mean=float(np.mean(top[:3])) if top else np.nan, wholeday_thr=res['thr'] if res else np.nan, day_mad=res['mad'] if res else np.nan,
               passes_wholeday=bool(res and pk >= res['thr']), peak_cc=pk, peak_offset_s=((j - i) * C.STEP / C.FS) if res else np.nan,
               second_peak_day=second, prominence=(pk / max(second, 1e-6)) if res else np.nan, n_live_channels=res['n_channels'] if res else 0)
    return row, ch, res


def facts():
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    c = catalog(); sh = S.load_shelly(); cusp = list(sh['cuspid'].astype(int))
    stations = [str(s) for s in sh['stations']]; prefixes = [str(p) for p in sh['channels']]
    # 1. picks for every detection
    p = keyed(pd.read_csv(PICKS_RUN, dtype={'origin': str})).drop_duplicates('key').set_index('key')
    rows = []
    for r in c.itertuples():
        if r.key in p.index:
            rows.append(dict(origin=r.origin, template_cuspid=int(r.template_cuspid), **{k: p.at[r.key, k] for k in PICK_COLS}, source='per-run table'))
        else:
            e = V.evidence(r.origin, r.template_cuspid); e.pop('origin'); e.pop('template_cuspid')
            rows.append(dict(origin=r.origin, template_cuspid=int(r.template_cuspid), **{k: e[k] for k in PICK_COLS}, source='computed for the final catalog'))
    pk = pd.DataFrame(rows); pk.to_csv(os.path.join(OUT, 'detection_picks.csv'), index=False)
    print(f'picks: {len(pk)} rows, {int((pk.source != "per-run table").sum())} computed', flush=True)
    # 2. channels and similarity for every detection that is not its template's own event
    ns = c[~c.self_detection].copy()
    chr_ = keyed(pd.read_csv(CH_RUN, dtype={'origin': str})); sm = keyed(pd.read_csv(SUM_RUN, dtype={'origin': str})).drop_duplicates('key').set_index('key')
    have = set(chr_.key) & set(sm.index); todo = ns[~ns.key.isin(have)]
    chan = [chr_[chr_.key.isin(set(ns.key))][['origin', 'template_cuspid', 'station', 'comp', 'phase', 'cc', 'weight']].assign(source='per-run table')]
    sim = [dict(origin=r.origin, template_cuspid=int(r.template_cuspid), **{k: sm.at[r.key, k] for k in SIM_COLS}, source='per-run table')
           for r in ns[ns.key.isin(have)].itertuples()]
    todo = todo.assign(day=pd.to_datetime(todo.origin).dt.strftime('%Y-%j'))
    for d, g in todo.groupby('day', sort=True):
        y, doy = int(d[:4]), int(d[5:]); day = S.scan.load_day(stations, prefixes, y, doy)
        for r in g.itertuples():
            ti = cusp.index(int(r.template_cuspid)); t1 = time.time()
            row, ch, _ = similarity_row(sh, ti, r.origin, day)
            sim.append(dict(origin=r.origin, template_cuspid=int(r.template_cuspid), **row, source='computed for the final catalog'))
            # per-channel rows as channel_cc.py writes them (detection_vs_template without the day mask)
            ch2, _, _ = S.detection_vs_template(sh, ti, r.origin)
            chan.append(pd.DataFrame([dict(origin=r.origin, template_cuspid=int(r.template_cuspid), station=o['station'], comp=o['comp'], phase=o['phase'],
                                           cc=o['cc'], weight=o.get('weight', np.nan)) for o in ch2]).assign(source='computed for the final catalog'))
            print(f"{r.origin[:22]} tpl {int(r.template_cuspid)}: net {row['net_cc']:.3f} (catalog {r.cc:.3f}) peak {row['peak_cc']:.3f} ch {row['n_channels']} | {time.time() - t1:.0f} s", flush=True)
        del day
    pd.DataFrame(sim).to_csv(os.path.join(OUT, 'detection_similarity.csv'), index=False)
    pd.concat(chan, ignore_index=True).to_csv(os.path.join(OUT, 'detection_channels.csv'), index=False)
    print(f'similarity: {len(sim)} rows ({len(todo)} computed); channels written; {time.time() - t0:.0f} s', flush=True)


def gallery(request):
    os.makedirs(FIG, exist_ok=True); t0 = time.time()
    c = catalog().set_index('key'); q = pd.read_csv(request, dtype={'origin': str}); q = keyed(q)
    sh = S.load_shelly(); cusp = list(sh['cuspid'].astype(int)); stations = [str(s) for s in sh['stations']]; prefixes = [str(p) for p in sh['channels']]
    q['day'] = pd.to_datetime(q.origin).dt.strftime('%Y-%j'); done = []
    for d, g in q.groupby('day', sort=True):
        y, doy = int(d[:4]), int(d[5:]); day = S.scan.load_day(stations, prefixes, y, doy)
        for r in g.itertuples():
            k = c.loc[r.key]; ti = cusp.index(int(r.template_cuspid)); o = pd.Timestamp(r.origin)
            row, ch, res = similarity_row(sh, ti, r.origin, day)
            lab = (f"reference event {k.reference_event_id}, detected by another event's template" if bool(k.is_reference) else 'new (not a reference event)')
            stem = os.path.join(FIG, f'{o:%Y%m%d_%H%M%S}_{int(r.template_cuspid)}')
            S.plot_overlay(ch, f"Template {int(r.template_cuspid)} vs detected waveform ({o:%Y-%m-%d %H:%M:%S} UTC, network CC {row['net_cc']:.2f}, catalog CC {k.cc:.2f}) — {lab}", stem + '_overlay.png')
            if res is not None:
                S.plot_cc_trace(res, sh, ti, r.origin, f"{o:%Y-%m-%d %H:%M:%S} UTC | template {int(r.template_cuspid)} | catalog CC {k.cc:.2f} | {int(k.n_present)} present | {int(k.n_templates)} templates | {lab}", stem + '_cc.png')
            done.append(dict(origin=r.origin, template_cuspid=int(r.template_cuspid), figure=os.path.relpath(stem, HERE), overlay=True, cc_trace=res is not None))
            print(f'{r.origin[:22]} {int(r.template_cuspid)} -> {stem}', flush=True)
        del day
    pd.DataFrame(done).to_csv(os.path.join(FIG, 'gallery_figures.csv'), index=False)
    print(f'gallery: {len(done)} detections, {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--gallery', help='request csv (origin, template_cuspid, label)'); a = ap.parse_args()
    gallery(a.gallery) if a.gallery else facts()
