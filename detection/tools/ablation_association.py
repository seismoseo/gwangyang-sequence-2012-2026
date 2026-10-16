#!/usr/bin/env python3
"""Association test for the R3 comparison (open item of the R3 document): why does the new run form events that the
old run's own data could have formed ("old data suffice")?

PyOcto is re-run on one year with three pick sets, with the pipeline's own association code and settings
(config.REGION_STRICT, kim1983, the same calls as core.run_association_year), writing only to the output folder:
  A  old       the old run's raw picks (its picks files) and its station table   -> should reproduce the old run's events
  B  kept      the new run's raw picks at stations the old run read with the same sensor on that day ("kept",
               the classes of compare_r3_old_new.py)                                -> the old run's data without the
                                                                                       picks the old run had at other sensors
  C  new       the new run's raw picks                                              -> should reproduce the new run's events
Each set's events are matched (origin time within 2 s, one to one) to the old run's and the new run's PyOcto events,
and to the new-only located events of the comparison (r3_new_only_events.csv, by cause). A against the old run and C
against the new run are the control: they show how exactly PyOcto reproduces a run.

  python tools/ablation_association.py --year 2021 [--sets A,B,C] [--threads 16]
Output: docs/stages/R3_detection/ablation/<year>_<set>_events.csv and ablation_<year>.csv (summary).
"""
import argparse, datetime as dt, glob, os, sys, tempfile, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "pipeline")); sys.path.insert(0, os.path.join(HERE, "tools"))
import config  # noqa: E402
import compare_r3_old_new as cmp  # noqa: E402   (match, pick_class, OLD_ROOT)

NEWP = os.path.join(HERE, "phasenet_plus_2020-2026")
OLDP = os.path.join(cmp.OLD_ROOT, "detection/phasenet_plus_2020-2026")
OUT = os.path.join(ROOT, "docs/stages/R3_detection/ablation")


def picks(root, y, d0=1, d1=366):
    fs = [f for f in sorted(glob.glob(os.path.join(root, f"detection_location/{y}/picks/picks_{y}.*.csv")))
          if d0 <= int(f.split(".")[-2]) <= d1]
    p = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    parts = p.station.str.split(".", expand=True); p["net"], p["code"] = parts[0], parts[1]
    return p


def associate(p, stations, threads):
    """core.run_association_year's PyOcto call on a given pick set and station table (Network, Code, Latitude, ...)"""
    import pyocto
    st = stations[stations.apply(lambda r: (r.Network, r.Code) in set(zip(p.net, p.code)), axis=1)]
    sp = st.assign(id=st.Network + "." + st.Code + ".").rename(columns={"Longitude": "longitude", "Latitude": "latitude", "Elevation": "elevation"})
    sp = sp[["id", "longitude", "latitude", "elevation"]].drop_duplicates("id").reset_index(drop=True)
    pk = p.rename(columns={"peak_time": "time"}).copy()
    pk["time"] = pd.to_datetime(pk["time"]).dt.tz_localize(None)
    pk["station"] = pk.net + "." + pk.code + "."
    pk = pk[pk.station.isin(sp.id)][["station", "phase", "time"]].copy()
    layers = pd.read_csv(config.VELOCITY_CSV)
    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, "vel_model")
        pyocto.VelocityModel1D.create_model(layers, 1.0, 100, 100, mp)
        vm = pyocto.VelocityModel1D(mp, tolerance=1.0)
        region = dict(config.REGION_STRICT, n_threads=threads)
        a = pyocto.OctoAssociator.from_area(velocity_model=vm, **region)
        a.transform_stations(sp)
        pk["time"] = pk["time"].apply(lambda x: x.timestamp())
        ev, asg = a.associate(pk, sp)
        a.transform_events(ev)
    if len(ev):
        ev["time"] = ev["time"].apply(dt.datetime.fromtimestamp, tz=dt.timezone.utc)
    ev["time"] = pd.to_datetime(ev["time"]).dt.tz_localize(None) if len(ev) else ev.get("time")
    return ev.sort_values("time").reset_index(drop=True) if len(ev) else ev


def n_matched(a, b, tol=cmp.MATCH_S):
    return len(cmp.match(a.reset_index(drop=True), b.reset_index(drop=True), tol)) if len(a) and len(b) else 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--year", type=int, required=True); ap.add_argument("--sets", default="A,B,C")
    ap.add_argument("--threads", type=int, default=16); ap.add_argument("--days", default=None, help="D0-D1: a test on these days only")
    a = ap.parse_args(); y = a.year
    d0, d1 = (int(x) for x in a.days.split("-")) if a.days else (1, 366)
    inwin = lambda t: (pd.to_datetime(t).dt.dayofyear >= d0) & (pd.to_datetime(t).dt.dayofyear <= d1)
    os.makedirs(OUT, exist_ok=True)
    st_new = pd.read_csv(cmp.ST_R3)
    st_old = pd.read_csv(os.path.join(OLDP, f"station_table/stations_{y}.csv"))
    old_codes = {(r.Network, r.Code) for r in pd.read_csv(os.path.join(HERE, "station_table/stations_pre_renewal.csv")).itertuples()}
    sensor_new = {(r.Network, r.Code): r.Sensor for r in st_new.itertuples()}
    ref = {}
    for name, root in (("old run", OLDP), ("new run", NEWP)):
        e = pd.read_csv(os.path.join(root, f"pyocto/pyocto_kim1983_{y}.csv")); e["time"] = pd.to_datetime(e.time).dt.tz_localize(None)
        ref[name] = e[inwin(e.time)].sort_values("time").reset_index(drop=True)
    no = pd.read_csv(os.path.join(ROOT, "docs/stages/R3_detection/tables/r3_new_only_events.csv"), parse_dates=["time"])
    no = no[(no.time.dt.year == y) & inwin(no.time)]
    groups = {c: no[no.cause == c].sort_values("time") for c in no.cause.unique()}
    rows = []
    for s in a.sets.split(","):
        t0 = time.time()
        if s == "A":
            p, stations = picks(OLDP, y, d0, d1), st_old
        elif s == "B":
            p = picks(NEWP, y, d0, d1); doy = pd.to_datetime(p.peak_time).dt.dayofyear
            p["cls"] = [cmp.pick_class(n, c, y, int(d), sensor_new.get((n, c)), old_codes) for n, c, d in zip(p.net, p.code, doy)]
            p, stations = p[p.cls == "kept"], st_new
        else:
            p, stations = picks(NEWP, y, d0, d1), st_new
        ev = associate(p, stations, a.threads)
        tag = f"{y}" + (f"_d{d0}-{d1}" if a.days else "")
        ev.to_csv(os.path.join(OUT, f"{tag}_{s}_events.csv"), index=False)
        r = dict(year=y, set=s, picks=len(p), events=len(ev), minutes=round((time.time() - t0) / 60, 1),
                 matched_old_run=n_matched(ev, ref["old run"]), matched_new_run=n_matched(ev, ref["new run"]))
        for c, g in groups.items():
            r[f"new_only[{c}]"] = f"{n_matched(g, ev, cmp.LINK_S)}/{len(g)}"
        rows.append(r); print(r, flush=True)
    out = pd.DataFrame(rows)
    out.assign(old_run_events=len(ref["old run"]), new_run_events=len(ref["new run"])).to_csv(os.path.join(OUT, f"ablation_{tag}.csv"), index=False)
    pd.set_option("display.width", 250); print(out.to_string(index=False))


if __name__ == "__main__":
    main()
