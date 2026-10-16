#!/usr/bin/env python3
"""R4 check (renewal plan, step R4): S differential times of the old run (better of raw N and E) against the renewal
(T after the time-dependent orientation correction), at a steady station (HAWA) and a re-oriented one (GOCB.HH, turned on
2020-05-24 inside the 2020 sequence), on the event pairs both runs correlated.

Events are matched old <-> new by origin time (DL ids within 2 s, one to one). For each common pair whose per-pair S file
exists in both runs, the station's (dt, cc) are compared. A pair is 'straddling' at GOCB when its two events fall on
different sides of 2020-05-24. Reported per station and class: pairs compared, pairs with cc >= 0.7 in each run, the median
cc in each run, and for pairs >= 0.7 in both the median |dt_new - dt_old|, raw and after removing the pair's origin-time
term (median P dt change over the stations, P being measured on Z in both runs).

  python relocation/code/r4_check_s_component.py [--new-root relocation/hypodd_dl_renewal_R4] [--stations HAWA,GOCB]
Output: docs/stages/R4_relocation/tables/s_component_check.csv (+ the per-pair rows in s_component_pairs.csv)
"""
import argparse, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OLD = os.path.join(ROOT, "relocation/hypodd_dl_2020-2026")
OUT = os.path.join(ROOT, "docs/stages/R4_relocation/tables")
GOCB_TURN = pd.Timestamp("2020-05-24")


def ids(root):
    m = pd.read_csv(os.path.join(root, "event_manifest.csv"), dtype={"event_id": str})
    return pd.DataFrame(dict(event_id=m.event_id, t=pd.to_datetime(m.event_id, format="%Y%m%d%H%M%S")))


def match(new_root, old_root=OLD, tol_s=2.0):
    """new event_id -> old event_id, one to one by origin time (closest first, within tol_s)"""
    o, n = ids(old_root), ids(new_root)
    cand = []
    for i, t in enumerate(n.t):
        d = (o.t - t).abs().dt.total_seconds()
        for j in np.where(d <= tol_s)[0]:
            cand.append((d[j], i, j))
    used_n, used_o, pairs = set(), set(), {}
    for d, i, j in sorted(cand):
        if i not in used_n and j not in used_o:
            used_n.add(i); used_o.add(j); pairs[n.event_id[i]] = o.event_id[j]
    print(f"events matched new<->old: {len(pairs)} (new {len(n)}, old {len(o)})")
    return pairs


def read_pair(path, stations=None):
    out = {}
    if not os.path.exists(path):
        return None
    for line in open(path):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 4 and (stations is None or p[0][2:] in stations):
            out[p[0][2:]] = (float(p[1]), float(p[2]))
    return out


def origin_term(f_new, f_old, sign):
    """the pair's origin-time term: median over stations of (dt_new - dt_old) on P (Z, measured the same way in both
    runs), stations with cc >= 0.7 in both; the catalog origin times differ between the runs, the measurement should not"""
    pn, po = read_pair(f_new), read_pair(f_old)
    if not pn or not po:
        return np.nan, 0
    d = [pn[k][0] - sign * po[k][0] for k in pn if k in po and pn[k][1] >= 0.7 and po[k][1] >= 0.7]
    return (float(np.median(d)), len(d)) if d else (np.nan, 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--new-root", default=os.path.join(ROOT, "relocation/hypodd_dl_renewal_R4"))
    ap.add_argument("--stations", default="HAWA,GOCB"); a = ap.parse_args()
    stations = a.stations.split(",")
    pairs = match(a.new_root, OLD)
    dS_new = os.path.join(a.new_root, "2.HypoDD/02.dt.cc/dt.cc_S"); dS_old = os.path.join(OLD, "2.HypoDD/02.dt.cc/dt.cc_S")
    rows = []
    for f in os.listdir(dS_new):
        e1, e2 = f.split("_")[-2:]
        if e1 not in pairs or e2 not in pairs:
            continue
        o1, o2 = pairs[e1], pairs[e2]
        sign = 1.0
        po = os.path.join(dS_old, f"dt.cc_S_{o1}_{o2}")
        if not os.path.exists(po):
            po = os.path.join(dS_old, f"dt.cc_S_{o2}_{o1}"); sign = -1.0
        old_v = read_pair(po, stations)
        if old_v is None:
            continue
        new_v = read_pair(os.path.join(dS_new, f), stations)
        fpn = os.path.join(a.new_root, "2.HypoDD/02.dt.cc/dt.cc_P", f.replace("dt.cc_S_", "dt.cc_P_"))
        fpo = po.replace("/dt.cc_S/", "/dt.cc_P/").replace("dt.cc_S_", "dt.cc_P_")
        oterm, n_p = origin_term(fpn, fpo, sign)
        t1, t2 = pd.to_datetime(e1, format="%Y%m%d%H%M%S"), pd.to_datetime(e2, format="%Y%m%d%H%M%S")
        for s in stations:
            if s not in old_v and s not in new_v:
                continue
            cls = ("straddling 2020-05-24" if s == "GOCB" and ((t1 < GOCB_TURN) != (t2 < GOCB_TURN)) else "same period")
            od = old_v.get(s, (np.nan, np.nan)); nd = new_v.get(s, (np.nan, np.nan))
            rows.append(dict(station=s, cls=cls, new_pair=f, dt_old=sign * od[0], cc_old=od[1], dt_new=nd[0], cc_new=nd[1],
                             origin_term=oterm, n_p_stations=n_p))
    d = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True); d.to_csv(os.path.join(OUT, "s_component_pairs.csv"), index=False)
    summ = []
    for (s, c), g in d.groupby(["station", "cls"]):
        both = g[(g.cc_old >= 0.7) & (g.cc_new >= 0.7)]
        summ.append(dict(station=s, pairs=cls_or(c), compared=len(g), old_ge07=int((g.cc_old >= 0.7).sum()), new_ge07=int((g.cc_new >= 0.7).sum()),
                         cc_old_median=round(float(g.cc_old.median()), 3), cc_new_median=round(float(g.cc_new.median()), 3),
                         both_ge07=len(both), dt_diff_median_s=round(float((both.dt_new - both.dt_old).abs().median()), 4) if len(both) else np.nan,
                         dt_diff_after_origin_term_median_s=round(float((both.dt_new - both.dt_old - both.origin_term).abs().median()), 4) if both.origin_term.notna().any() else np.nan,
                         within_5ms_after_origin_term=round(float(((both.dt_new - both.dt_old - both.origin_term).abs() <= 0.005).mean()), 3) if both.origin_term.notna().any() else np.nan))
    S = pd.DataFrame(summ); S.to_csv(os.path.join(OUT, "s_component_check.csv"), index=False)
    pd.set_option("display.width", 200); print(S.to_string(index=False))


def cls_or(c):
    return c


if __name__ == "__main__":
    main()
