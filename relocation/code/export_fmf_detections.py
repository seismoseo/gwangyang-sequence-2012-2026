#!/usr/bin/env python
"""GWANGYANG G8 (port of Imsil reloc/export_fmf_detections.py, decision D22): run root for the joint relocation of the
template-matching detections with the reference events, on the renewal layout and with R/T exports (S on T).

Real-pick events: the focused 25 m reference set (catalogs/reference_25m/focused_reference.csv = the templates). Their SAC
bundles (PhaseNet+ picks in `a`/`t0`, with R/T) are COPIED from the G5 run root (registry reloc_final_run/waveforms_100km):
the framework re-references SAC headers in place, so the G5 run is never touched.
Detections: every new detection (is_new) of the G7 catalog. Each inherits its representative template's picks shifted to the
detection origin:
    pick(detection, station, phase) = pick(template event, station, phase) + (detection origin - template origin)
with the template's absolute picks read from its SAC headers (reference time + a / t0, as template_matching/templates.py
reads them) and the template origin from the template table (registry tm_templates_195). The SAC bundle is cut from the
continuous link tree as event_sac_export.export_event cuts G5 events (same window, the station table of the detection's
year, the same channel policy, R/T from registry/orientation.py at the detection origin), with the template's relocated
position as the initial location.
Differences from Imsil: registry paths; reference ids in column event_id; the station table of each detection's own year
(Imsil: 2025 for all); R and T written (event_sac_export.ROTATE_T), so run_fmf_hypodd.py measures S on T as G5 did.
Writes <run_root>/waveforms_100km/<event_id>/ and <run_root>/fmf_relocation_input.csv.
DRY RUN BY DEFAULT (prints the plan); --apply writes.
    python relocation/code/export_fmf_detections.py [--run-root relocation/hypodd_fmf_2012-2026] [--apply]
"""
import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # the project registry, found upward from this file
import argparse
import glob
import os
import shutil

import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, str(haenam.ROOT / "detection/pipeline"))
import config as pipe_config                                   # noqa: E402  (DL pipeline config)
import event_sac_export as ese                                 # noqa: E402

FMF_CSV = str(haenam.ROOT / "template_matching/catalogs/fmf_haenam_catalog_G7_5_20_seg.csv")   # the G7 catalog (run_G7.sh catalog)
TPL_CSV = haenam.path(None, "tm_templates_195")
REF_CSV = os.path.join(haenam.path(None, "reference_25m_dir"), "focused_reference.csv")
SRC_RUN = haenam.path(None, "reloc_final_run")


def template_picks(event_id):
    """{(net, sta): {'P': UTCDateTime, 'S': UTCDateTime}} from the template event's SAC headers (reference time + a / t0)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(SRC_RUN, "waveforms_100km", event_id, "*.sac"))):
        h = obspy.read(f, headonly=True)[0].stats; s = h.sac
        ref = UTCDateTime(year=s.nzyear, julday=s.nzjday, hour=s.nzhour, minute=s.nzmin, second=s.nzsec, microsecond=int(s.nzmsec) * 1000)
        d = out.setdefault((h.network, h.station), {})
        if "a" in s and s.a > -1e4: d.setdefault("P", ref + float(s.a))
        if "t0" in s and s.t0 > -1e4: d.setdefault("S", ref + float(s.t0))
    return {k: v for k, v in out.items() if v}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default=str(haenam.ROOT / "relocation/hypodd_fmf_2012-2026"))
    ap.add_argument("--fmf-csv", default=FMF_CSV)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    ese.ROTATE_T = True
    fmf = pd.read_csv(a.fmf_csv, parse_dates=["origin"]); new = fmf[fmf.is_new].sort_values("origin").reset_index(drop=True)
    tpl = pd.read_csv(TPL_CSV, dtype={"event_id": str}, parse_dates=["origin_time_UTC"]).set_index("cuspid")
    ref = pd.read_csv(REF_CSV, dtype={"event_id": str})
    ref_ids = sorted(ref.event_id.astype(str))
    tp_cache = {}
    rows = []
    for r in new.itertuples():
        o = UTCDateTime(pd.Timestamp(r.origin).to_pydatetime()); eid = ese.event_id_from_time(o)
        t = tpl.loc[int(r.template_cuspid)]; te = str(t.event_id)
        if te not in tp_cache: tp_cache[te] = template_picks(te)
        delta = o - UTCDateTime(pd.Timestamp(t.origin_time_UTC).to_pydatetime())
        picks = {k: {ph: (tt + delta) - o for ph, tt in v.items()} for k, v in tp_cache[te].items()}
        rows.append(dict(kind="detection", event_id=eid, origin=str(o)[:23], lat=float(t.latitude), lon=float(t.longitude), depth=float(t.depth_km),
                         template_cuspid=int(r.template_cuspid), template_event_id=te, fmf_cc=float(r.cc), n_present=int(r.n_present),
                         n_P=sum("P" in v for v in picks.values()), n_S=sum("S" in v for v in picks.values()), picks_rel=picks, o_utc=o))
    det = pd.DataFrame(rows)
    if not len(det):
        raise SystemExit(f"no new detections in {a.fmf_csv}")
    clash = set(det.event_id) & set(ref_ids); dup = det.event_id[det.event_id.duplicated()].tolist()
    if clash or dup:
        raise SystemExit(f"event-id collisions: with references {sorted(clash)}; among detections {dup}")
    print(f"[plan] run root {a.run_root}")
    print(f"[plan] {len(ref_ids)} reference events (SAC copied from {SRC_RUN}/waveforms_100km)")
    print(f"[plan] {len(det)} detections from {len(det.template_event_id.unique())} templates; inherited picks per detection: "
          f"P median {int(det.n_P.median())} (min {det.n_P.min()}), S median {int(det.n_S.median())} (min {det.n_S.min()})")
    print("[plan] detections per year:", det.o_utc.map(lambda u: u.year).value_counts().sort_index().to_dict())
    if not a.apply:
        print("(dry run - add --apply to copy the reference bundles, export the detections and write fmf_relocation_input.csv)"); return
    wf = os.path.join(a.run_root, "waveforms_100km"); os.makedirs(wf, exist_ok=True)
    for eid in ref_ids:
        src, dst = os.path.join(SRC_RUN, "waveforms_100km", eid), os.path.join(wf, eid)
        if not os.path.isdir(dst): shutil.copytree(src, dst)
    stations_for_year = {}
    n_done = n_rt = n_unknown = 0
    for r in det.itertuples():
        out_dir = os.path.join(wf, r.event_id)
        if os.path.isdir(out_dir) and any(f.endswith(".sac") for f in os.listdir(out_dir)): continue
        y = r.o_utc.year
        if y not in stations_for_year:
            stations_for_year[y] = ese.load_stations_for_year(pipe_config.station_table_dir("phasenet_plus"), y)
        ev = ese.EventMeta(r.event_id, r.o_utc, r.lat, r.lon, r.depth)
        for sr in stations_for_year[y].itertuples():
            sm = ese.StationMeta(sr.Network, sr.Code, float(sr.Latitude), float(sr.Longitude), float(getattr(sr, "Elevation", np.nan)))
            st = ese.read_continuous_for_event(pipe_config.CONTINUOUS, sm.network, sm.code, r.o_utc)
            if not len(st): continue
            st, rt = ese.add_rt(st, sm, r.o_utc)
            n_rt += rt == "rt"; n_unknown += rt == "unknown"
            pk = r.picks_rel.get((sm.network, sm.code), {})
            ese.write_event_sac(st, out_dir, event_meta=ev, station_meta=sm, p_rel_s=pk.get("P"), s_rel_s=pk.get("S"))
        n_done += 1
        if n_done % 25 == 0: print(f"  exported {n_done} detections", flush=True)
    refrows = pd.DataFrame(dict(kind="reference", event_id=ref_ids))
    table = pd.concat([refrows, det.drop(columns=["picks_rel", "o_utc"])], ignore_index=True)
    table.to_csv(os.path.join(a.run_root, "fmf_relocation_input.csv"), index=False)
    print(f"[export] {len(ref_ids)} reference bundles copied, {n_done} detection bundles written -> {wf}; "
          f"R/T written for {n_rt} station-events, orientation unknown for {n_unknown}; table -> {a.run_root}/fmf_relocation_input.csv")


if __name__ == "__main__":
    main()
