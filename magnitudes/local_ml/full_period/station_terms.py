"""Renewal R6 (ISSUES #42, user 2026-10-14): one constant ML term per station and response epoch, with a held-out check.

A station whose ML is offset from KMA's by the same amount on every event is corrected, not excluded. For every station and
response epoch the term is minus the median of (station ML - KMA ML) over the KMA-anchored events; it is added to that
station's ML on every event of the epoch, and the event ML is recomputed exactly as full_magnitudes.py does (median over the
same stations, >= 3 picked stations for the primary value). No station is dropped and no other gate is added.

Check (before adoption): for each anchor event, terms are re-estimated without that event and its event ML recomputed; the
held-out corrected ML and the uncorrected ML are compared with KMA ML on the same events.

  python station_terms.py                 # writes station_terms/ next to this file (reads full_magnitudes.py outputs)

Inputs: station_magnitudes.csv and full_period_ML.csv (full_magnitudes.py), responses_full_period.xml + data/jnu/JU_station.xml
(the response epochs), registry/streams.csv (each station's sensor), the relocated catalog (anchors).
"""
import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam
from pathlib import Path
import json, hashlib, warnings
import numpy as np
import pandas as pd
import obspy
sys.path.insert(0, str(haenam.ROOT / 'magnitudes/local_ml'))
import buan_sheen2018_snapshot as S

HERE = Path(__file__).resolve().parent
OUT = HERE / 'station_terms'
CATALOG = Path(haenam.path(None, 'dl_relocated_full'))
REFERENCE = Path(haenam.path(None, 'reference_25m_csv'))   # the R4 25 m reference (646; registry)
STREAMS = haenam.ROOT / 'registry/streams.csv'
INVENTORIES = [HERE / 'responses_full_period.xml', haenam.ROOT / 'data/jnu/JU_station.xml']

# ---- parameters (all of them) ----
MIN_ANCHORS = 10      # a station-epoch with fewer KMA-anchored measurements gets no term (term 0, flagged)
REF_FREQ = 5.0        # Hz; response amplitude compared between consecutive inventory epochs (inside the 0.5-10 Hz band)
EPOCH_TOL = 0.01      # consecutive epochs whose response amplitude at REF_FREQ differs by < 1 % are one response epoch
ERA_B1, ERA_B2 = '2020-08-01', '2026-08-14'   # era labels for the report only (as in nb33)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def anchors(cat):
    """KMA ML of the relocated events, as full_period_publication.load defines known_KMA_ML: the matched template's ML where
    the match is valid and unique among the relocated events; restricted to the 25 m reference events (as nb33)."""
    valid = cat.template_match.notna() & cat.template_match.ne('-')
    dup = valid & cat.template_match.duplicated(keep=False)
    ref = pd.read_csv(REFERENCE, dtype={'event_id': str}).event_id
    a = cat[valid & ~dup & cat.event_id.isin(ref)].copy()
    return a[['event_id', 'ML_template']].rename(columns={'event_id': 'eid', 'ML_template': 'kma_ml'})


def response_epochs():
    """(net, sta) -> list of (start, end, label): Z-channel response epochs of the station's chosen sensor, consecutive
    inventory epochs merged where the response amplitude at REF_FREQ changes by less than EPOCH_TOL."""
    streams = pd.read_csv(STREAMS)
    sensor = {(r.net, r.sta): r.sensor for r in streams.itertuples()}
    out = {}
    for path in INVENTORIES:
        inv = obspy.read_inventory(str(path))
        for net in inv:
            for sta in net:
                band = sensor.get((net.code, sta.code))
                chans = [c for c in sta if c.code.endswith('Z') and (band is None or c.code.startswith(band)) and c.response is not None]
                if band is None:                     # a code outside the stream table: its only band, if it has one
                    bands = {c.code[:2] for c in chans}
                    if len(bands) != 1:
                        continue
                rows = []
                for c in sorted(chans, key=lambda c: c.start_date):
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        amp = float(abs(c.response.get_evalresp_response_for_frequencies([REF_FREQ], output='VEL')[0]))
                    rows.append([c.start_date, c.end_date, amp])
                merged = []
                for s, e, amp in rows:
                    if merged and abs(amp / merged[-1][2] - 1) < EPOCH_TOL:
                        merged[-1][1] = e
                    else:
                        merged.append([s, e, amp])
                if merged:
                    out[(net.code, sta.code)] = [(s, e, f"{net.code}.{sta.code} from {str(s)[:10]}") for s, e, _ in merged]
    return out


def epoch_of(ep, net, sta, t):
    for s, e, lab in ep.get((net, sta), []):
        if t >= s and (e is None or t < e):
            return lab
    return f"{net}.{sta} (no epoch)"


def event_tables(st, ev):
    """Event ML from station values, as full_magnitudes.event(): S.event_ml for the primary value, the plain median for
    ml_min1/ml_min2 (gated on picked stations)."""
    rows = []
    for eid, tab in st.groupby('eid', sort=False):
        agg = S.event_ml(tab)
        ml_sta = tab.ml_h.where(tab.ml_h.notna(), tab.ml_v)
        med = float(ml_sta.median()) if ml_sta.notna().any() else np.nan
        rows.append(dict(eid=eid, ml_min1=med if agg['n_pick'] >= 1 else np.nan, ml_min2=med if agg['n_pick'] >= 2 else np.nan,
                         ml_min3=agg['ml'], n_pick=agg['n_pick'], n_sta=agg['n_sta']))
    t = pd.DataFrame(rows)
    return ev[['eid']].merge(t, on='eid', how='left')


def terms_from(res):
    g = res.groupby(['network', 'station', 'epoch']).residual
    t = g.agg(n='size', median='median').reset_index()
    t['mad'] = g.agg(lambda r: float(1.4826 * np.median(np.abs(r - np.median(r))))).to_numpy()
    t['term'] = np.where(t.n >= MIN_ANCHORS, -t['median'], 0.0)
    t['applied'] = t.n >= MIN_ANCHORS
    return t


def apply_terms(st, terms):
    key = terms.set_index(['network', 'station', 'epoch']).term
    k = pd.MultiIndex.from_frame(st[['network', 'station', 'epoch']])
    term = pd.Series(key.reindex(k).to_numpy(), index=st.index).fillna(0.0)
    out = st.copy()
    out['term'] = term
    out['ml_h'] = st.ml_h + term
    out['ml_v'] = st.ml_v + term
    return out


def main():
    OUT.mkdir(exist_ok=True)
    st = pd.read_csv(HERE / 'station_magnitudes.csv', dtype={'eid': str})
    ev = pd.read_csv(HERE / 'full_period_ML.csv', dtype={'eid': str, 'event_id': str, 'template_match': str}, parse_dates=['origin_time'])
    cat = pd.read_csv(CATALOG, dtype={'event_id': str, 'template_match': str})
    assert set(ev.eid) == set(cat.event_id)
    ep = response_epochs()
    tmap = dict(zip(ev.eid, ev.origin_time))
    st['epoch'] = [epoch_of(ep, n, s, obspy.UTCDateTime(tmap[e])) for n, s, e in zip(st.network, st.station, st.eid)]
    st['ml_sta'] = st.ml_h.where(st.ml_h.notna(), st.ml_v)
    a = anchors(cat)
    res = st[st.ml_sta.notna()].merge(a, on='eid')
    res['residual'] = res.ml_sta - res.kma_ml
    res['era'] = np.select([res.eid.map(tmap).lt(ERA_B1), res.eid.map(tmap).lt(ERA_B2)], ['2020 period', 'Quiescence'], '2026 sequence')

    terms = terms_from(res)
    corrected_st = apply_terms(st.drop(columns='ml_sta'), terms)
    base = event_tables(st.drop(columns=['ml_sta', 'epoch']), ev)
    corr = event_tables(corrected_st, ev)

    # held-out check: each anchor event's terms re-estimated without it
    held = []
    for eid in a.eid[a.eid.isin(st.eid)]:
        t = terms_from(res[res.eid != eid])
        one = apply_terms(st[st.eid == eid].drop(columns='ml_sta'), t)
        r = event_tables(one, ev[ev.eid == eid]).iloc[0]
        held.append(dict(eid=eid, ml_min1_heldout=r.ml_min1, ml_min3_heldout=r.ml_min3))
    held = pd.DataFrame(held)
    v = a.merge(base.rename(columns={c: c + '_uncorrected' for c in ['ml_min1', 'ml_min2', 'ml_min3']})[['eid', 'ml_min1_uncorrected', 'ml_min3_uncorrected']], on='eid')
    v = v.merge(held, on='eid').merge(corr[['eid', 'ml_min1', 'ml_min3']].rename(columns={'ml_min1': 'ml_min1_corrected', 'ml_min3': 'ml_min3_corrected'}), on='eid')
    rows = []
    for level in ['ml_min3', 'ml_min1']:
        cols = [f'{level}_uncorrected', f'{level}_heldout', f'{level}_corrected']
        common = v.dropna(subset=cols)
        for c in cols:
            e = common[c] - common.kma_ml
            rows.append(dict(level=level, magnitude=c.split('_')[-1], n=len(e), median=float(e.median()), RMS=float(np.sqrt(np.mean(e ** 2))),
                             MAD=float(1.4826 * np.median(np.abs(e - e.median()))), q90_abs=float(e.abs().quantile(.9))))
    check = pd.DataFrame(rows)

    terms.to_csv(OUT / 'station_terms.csv', index=False)
    res.to_csv(OUT / 'anchor_residuals.csv', index=False)
    corrected_st.to_csv(OUT / 'station_magnitudes.csv', index=False)
    out = ev.drop(columns=['ml_min1', 'ml_min2', 'ml_min3']).merge(corr[['eid', 'ml_min1', 'ml_min2', 'ml_min3']], on='eid', validate='one_to_one')
    out.to_csv(OUT / 'full_period_ML.csv', index=False)
    v.to_csv(OUT / 'heldout_events.csv', index=False)
    check.to_csv(OUT / 'heldout_check.csv', index=False)
    srcs = [Path(__file__), HERE / 'station_magnitudes.csv', HERE / 'full_period_ML.csv', CATALOG, REFERENCE, STREAMS] + INVENTORIES
    (OUT / 'provenance.json').write_text(json.dumps(dict(
        status='NOT ADOPTED until the user has seen heldout_check.csv (ISSUES #42)',
        parameters=dict(MIN_ANCHORS=MIN_ANCHORS, REF_FREQ=REF_FREQ, EPOCH_TOL=EPOCH_TOL),
        n_anchor_events=int(a.eid.isin(st.eid).sum()), n_station_epochs=len(terms), n_terms_applied=int(terms.applied.sum()),
        inputs=[dict(path=str(p), sha256=sha(p)) for p in srcs]), indent=2))
    print(f"anchors {int(a.eid.isin(st.eid).sum())} events; station-epochs {len(terms)}, terms applied {int(terms.applied.sum())}")
    print(check.round(3).to_string(index=False))
    return terms, check


if __name__ == '__main__':
    main()
