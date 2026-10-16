#!/usr/bin/env python3
"""Renewal R3 against the old full-period run (27.Haenam_sequence, frozen): which located events are common, new-only and
old-only, and why the new run finds more.

Matching: a new and an old located event (HYPOINVERSE .sum, kim1983) are the same event when their origin times differ
by at most MATCH_S; pairs are formed one-to-one, closest first.

Explaining a new-only event: each of its association picks (PyOcto, before pick augmentation) is classed by what the
old run could read at that station ON THAT DAY, from the old run's own link tree (the old pipeline took the first of
HH, EL, HG present each day; the Sensor column of the old station table does not record that):
  kept           the old run read the same sensor that day
  other sensor   the old run read another sensor that day (e.g. HH where the renewal chose HG)
  new data       the old tree had no data for the station that day (days now linked, repaired or filled from KIGAM)
  added station  the station was not in the old run at all
If the kept picks alone do not meet the association minimum (config.REGION_STRICT: 3 P, 3 S, 6 picks, 2 stations with
both P and S), the event needs the renewal's additions. Otherwise the old run's own data would have been enough, and
the difference comes from association details (PyOcto is a streaming associator: other picks change its candidates).

Outputs (tables for the R3 stage document), into docs/stages/R3_detection/tables/:
  r3_old_new_by_year.csv          counts per year: new, old, common, new-only (split by cause), old-only; the same in the box
  r3_matched_location_change.csv  per matched event in the Haenam box: the epicentre and depth change
  r3_new_only_events.csv          every new-only located event with its pick split and cause
  r3_old_only_events.csv          every old-only located event

  python tools/compare_r3_old_new.py [--years 2020,2021]       (default: every year located in both runs)
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
OLD_ROOT = "/home/msseo/works/27.Haenam_sequence"
sys.path.insert(0, os.path.join(HERE, "pipeline"))
import config  # noqa: E402
from sumio import read_sum  # noqa: E402

NEW = os.path.join(HERE, "phasenet_plus_2020-2026")
OLD = os.path.join(OLD_ROOT, "detection/phasenet_plus_2020-2026")
OUT = os.path.join(ROOT, "docs/stages/R3_detection/tables")
OLD_TREE = os.path.join(OLD_ROOT, "data/continuous/sds_links")             # the old run's link tree (what it could read)
POLICY = ("HH", "EL", "HG")                                                # the pipeline's CHANNEL_POLICY order
MATCH_S = 2.0          # same event if the origin times differ by at most this
NEAR_S = 5.0           # reported, not matched: a new-only event with an old-only event within this
LINK_S = 3.0           # located event <-> its PyOcto event (HYPOINVERSE moves the origin time a little)
MIN = config.REGION_STRICT
# the R3 station table: R3x (run_r3x_detection.sh prepare) adds the 2018-2019 predecessor codes to stations.csv after
# saving R3's table as stations_r3.csv
ST_R3 = next(f for f in (os.path.join(HERE, "station_table/stations_r3.csv"), os.path.join(HERE, "station_table/stations.csv")) if os.path.exists(f))


def located(root, y):
    df = read_sum(os.path.join(root, f"HypoInv/kim1983/{config.REGION_NAME}{y}.sum"))
    df = df[df.time.notna() & df.lat.notna()].sort_values("time").reset_index(drop=True)
    b = config.HAENAM_BOX
    df["in_box"] = df.lat.between(b[0], b[1]) & df.lon.between(b[2], b[3])
    return df


def match(a, b, tol):
    """one-to-one pairs (i in a, j in b), closest in time first, |dt| <= tol"""
    ta = a.time.values.astype("datetime64[ns]").astype(np.int64) / 1e9
    tb = b.time.values.astype("datetime64[ns]").astype(np.int64) / 1e9
    cand = []
    for i, t in enumerate(ta):
        lo, hi = np.searchsorted(tb, t - tol), np.searchsorted(tb, t + tol, side="right")
        cand += [(abs(tb[j] - t), i, j) for j in range(lo, hi)]
    used_a, used_b, pairs = set(), set(), []
    for d, i, j in sorted(cand):
        if i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j); pairs.append((i, j, d))
    return pairs


_old_cache = {}
def old_band(net, sta, y, d):
    """the band the old run read at (net, sta) on day (y, d): the first of POLICY with a Z day file in the old tree, else None"""
    k = (net, sta, y, d)
    if k not in _old_cache:
        _old_cache[k] = next((b for b in POLICY if os.path.exists(os.path.join(OLD_TREE, net, sta, f"{b}Z.D", f"{net}.{sta}..{b}Z.D.{y}.{d:03d}"))), None)
    return _old_cache[k]


def pick_class(net, sta, y, d, sensor_new, old_codes):
    if (net, sta) not in old_codes:
        return "added station"
    ob = old_band(net, sta, y, d)
    return "new data" if ob is None else "kept" if ob == sensor_new else "other sensor"


def meets(p):
    """the association minimum (config.REGION_STRICT) for a set of picks (columns sta, phase)"""
    nP, nS = int((p.phase == "P").sum()), int((p.phase == "S").sum())
    both = len(set(p.sta[p.phase == "P"]) & set(p.sta[p.phase == "S"]))
    return nP >= MIN["n_p_picks"] and nS >= MIN["n_s_picks"] and nP + nS >= MIN["n_picks"] and both >= MIN["n_p_and_s_picks"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--years", default=None); a = ap.parse_args()
    yrs_new = {int(os.path.basename(f)[len(config.REGION_NAME):-4]) for f in glob.glob(f"{NEW}/HypoInv/kim1983/{config.REGION_NAME}*.sum")}
    yrs_old = {int(os.path.basename(f)[len(config.REGION_NAME):-4]) for f in glob.glob(f"{OLD}/HypoInv/kim1983/{config.REGION_NAME}*.sum")}
    years = sorted(int(y) for y in a.years.split(",")) if a.years else sorted(yrs_new & yrs_old)

    st_new = pd.read_csv(ST_R3)
    st_old = pd.read_csv(os.path.join(HERE, "station_table/stations_pre_renewal.csv"))
    old_codes = {(r.Network, r.Code) for r in st_old.itertuples()}
    sensor_new = {(r.Network, r.Code): r.Sensor for r in st_new.itertuples()}
    print(f"stations not in the old run: {', '.join(sorted(f'{n}.{s}' for n, s in set(sensor_new) - old_codes))}")

    rows, moved, new_only_all, old_only_all = [], [], [], []
    for y in years:
        n, o = located(NEW, y), located(OLD, y)
        pairs = match(n, o, MATCH_S)
        mi = {i for i, _, _ in pairs}; mj = {j for _, j, _ in pairs}
        n_only, o_only = n.drop(index=list(mi)), o.drop(index=list(mj))
        # association picks of the new-only events
        ev = pd.read_csv(f"{NEW}/pyocto/pyocto_kim1983_{y}.csv"); ev["time"] = pd.to_datetime(ev.time).dt.tz_localize(None)
        ev = ev.sort_values("time").reset_index(drop=True)
        asg = pd.read_csv(f"{NEW}/pyocto/pyocto_assignment_kim1983_{y}.before_augmentation.csv")
        asg["net"] = asg.station.str.split(".").str[0]; asg["sta"] = asg.station.str.split(".").str[1]
        by_ev = dict(tuple(asg.groupby("event_idx")))
        link = {i: ev.idx[j] for i, j, _ in match(n_only.reset_index(), ev, LINK_S)}
        causes = []
        for k, (ix, r) in enumerate(n_only.iterrows()):
            p = by_ev.get(link.get(k), pd.DataFrame(columns=asg.columns)).copy()
            doy = int(r.time.dayofyear)
            p["cls"] = [pick_class(nn, ss, y, doy, sensor_new.get((nn, ss)), old_codes) for nn, ss in zip(p.net, p.sta)]
            cause = ("no association picks found" if p.empty else
                     "needs the renewal's additions" if not meets(p[p.cls == "kept"]) else "old stations would suffice")
            causes.append(cause)
            nc = p.cls.value_counts()
            new_only_all.append(dict(time=r.time, lat=r.lat, lon=r.lon, depth=r.depth, in_box=r.in_box, num=r.num, rms=r.rms,
                                     picks=len(p), picks_kept=int(nc.get("kept", 0)), picks_other_sensor=int(nc.get("other sensor", 0)),
                                     picks_new_data=int(nc.get("new data", 0)), picks_added_station=int(nc.get("added station", 0)),
                                     not_kept_stations=" ".join(sorted(set(p.sta[p.cls != "kept"]))), cause=cause))
        causes = pd.Series(causes, index=n_only.index, dtype=object)
        # a new-only event with an old event 2-5 s away is probably the same event located with another origin time
        near5 = pd.Series([bool(len(match(pd.DataFrame(dict(time=[t])), o_only, NEAR_S))) for t in n_only.time], index=n_only.index)
        for k, ix in enumerate(n_only.index):
            new_only_all[len(new_only_all) - len(n_only) + k]["old_event_within_5s"] = bool(near5[ix])
        for r in o_only.itertuples():
            old_only_all.append(dict(time=r.time, lat=r.lat, lon=r.lon, depth=r.depth, in_box=r.in_box, num=r.num, rms=r.rms))
        for i, j, d in pairs:
            if n.in_box[i] or o.in_box[j]:
                dh = 111.2 * np.hypot(n.lat[i] - o.lat[j], (n.lon[i] - o.lon[j]) * np.cos(np.radians(o.lat[j])))
                moved.append(dict(year=y, time_new=n.time[i], dt_s=round(d, 2), dh_km=round(dh, 2), dz_km=round(n.depth[i] - o.depth[j], 2),
                                  num_new=n.num[i], num_old=o.num[j], rms_new=n.rms[i], rms_old=o.rms[j]))
        for scope, sel_n, sel_o in (("all", n.index, o.index), ("box", n.index[n.in_box], o.index[o.in_box])):
            nb, ob = set(sel_n), set(sel_o)
            com = sum(1 for i, j, _ in pairs if i in nb)
            no = n_only.index.intersection(list(nb))
            rows.append(dict(year=y, scope=scope, new=len(nb), old=len(ob), common=com, new_only=len(no),
                             new_only_needs_added=int((causes[no] == "needs the renewal's additions").sum()),
                             new_only_old_would_suffice=int((causes[no] == "old stations would suffice").sum()),
                             new_only_unlinked=int((causes[no] == "no association picks found").sum()),
                             new_only_old_within_5s=int(near5[no].sum()),
                             old_only=len(o_only.index.intersection(list(ob)))))
    os.makedirs(OUT, exist_ok=True)
    t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT, "r3_old_new_by_year.csv"), index=False)
    pd.DataFrame(moved).to_csv(os.path.join(OUT, "r3_matched_location_change.csv"), index=False)
    pd.DataFrame(new_only_all).to_csv(os.path.join(OUT, "r3_new_only_events.csv"), index=False)
    pd.DataFrame(old_only_all).to_csv(os.path.join(OUT, "r3_old_only_events.csv"), index=False)
    pd.set_option("display.width", 200)
    for scope in ("all", "box"):
        print(f"\n{scope}:"); print(t[t.scope == scope].drop(columns="scope").to_string(index=False))
    m = pd.DataFrame(moved)
    if len(m):
        print(f"\nmatched events in the box: {len(m)}; epicentre change median {m.dh_km.median():.2f} km "
              f"(90% {m.dh_km.quantile(.9):.2f}), depth change median {m.dz_km.abs().median():.2f} km")
    print("->", OUT)


if __name__ == "__main__":
    main()
