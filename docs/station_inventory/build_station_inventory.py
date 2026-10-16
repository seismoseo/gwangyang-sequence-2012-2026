#!/usr/bin/env python
# GWANGYANG PORT of 27.Haenam_sequence/docs/decisions/station_inventory_20261011/build_station_inventory.py (the Haenam renewal R1/R2
# inventory; copy kept next to this file as build_station_inventory_haenam.py). Same functions and rules; Haenam-only decisions,
# patches and added stations removed; region, window and sources changed:
#   - station set: every KS/KG code on MARA with data in the window within 100 km of the Gwangyang cluster (Haenam rule)
#     (data/candidate_stations_100km.csv; coordinates from the master StationXML, else 33's national orientation table;
#     NAHA's master coordinates are wrong and use 34.8167 N 127.9261 E, as the Haenam renewal found);
#   - KG codes that MARA also files under KS in 2017 (GRE, GSU, HDB, JRB, MUN, YSB) are duplicates: KG is used;
#   - temporary codes <SITE>T are merged into their site (Haenam rule), e.g. GWYBT -> GWYB (both 2023-01-12..07-03): the link tree
#     links a temporary code only on days its site's chosen sensor lacks (decision G-D7);
#   - sensor choice: the user's Haenam sensor rule (fewest discontinuities over the window; ties broadband > short-period >
#     accelerometer) for every station with more than one sensor; where any sensor's orientation was never measured the rule
#     cannot be applied and the current rule (broadband, else short-period, else accelerometer) stands, flagged.
# Read-only on every source; writes only into tables/ next to this file.
#   python docs/station_inventory/build_station_inventory.py
import glob, os, re, sys, warnings
from collections import defaultdict
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

G = "/home/msseo/works/73.Gwangyang_sequence"
HERE = os.path.dirname(os.path.abspath(__file__))
T33 = "/home/msseo/works/33.KS_station_orientation"
RESP = "/home/msseo/works/27.Haenam_sequence/data/stations/responses"        # master StationXML, KIGAM StationXML, NECIS RESP (read only)
CLUSTER = (35.0000, 127.7286)                                                 # gwangyang_pnplus relocation centroid
W0, W1 = "2012-12-20", "2026-10-06"                                           # first GWYB day .. last day on MARA (2026-10-15 listing)
SEQ = {"2021": ("2021-07-20", "2021-08-29"), "2026": ("2026-01-01", "2026-02-04")}
AVAIL = f"{G}/data/mara_availability_raw_20261015.txt"
CANDIDATES = f"{G}/data/candidate_stations_100km.csv"
MERGED = {}                                                                   # filled in stations(): every code <SITE>T whose site is in the set
KS_DUPLICATES_OF_KG = {"GRE", "GSU", "HDB", "JRB", "MUN", "YSB"}
COVER_SLACK_D = 365                                                           # GWANGYANG PORT (G-D1b): coverage-first slack
EXCLUDE_CODES = {"KWJ": "MARA holds 4 HH files per year 2012-2015 named DOY 000-003 by sequence, holding 01-01, 04-10, 07-19 and 10-27 at 80 Hz, plus one 80 Hz day in 2016: wrong dates and rate (G-D20)"}   # GWANGYANG PORT


def dist_az(lat, lon):
    from obspy.geodetics import gps2dist_azimuth
    d, az, _ = gps2dist_azimuth(CLUSTER[0], CLUSTER[1], lat, lon)
    return round(d / 1e3, 1), round(az)


def coordinates():
    import re as _re
    txt = open(f"{RESP}/master/KS_KG_metadata_1.0.2.xml").read(); xy = {}
    for m in _re.finditer(r'<Network code="(\w+)".*?</Network>', txt, _re.S):
        for s in _re.finditer(r'<Station code="(\w+)"[^>]*>\s*<Latitude[^>]*>([-\d.]+)</Latitude>\s*<Longitude[^>]*>([-\d.]+)</Longitude>\s*<Elevation[^>]*>([-\d.]+)</Elevation>', m.group(0), _re.S):
            xy.setdefault((m.group(1), s.group(1)), (float(s.group(2)), float(s.group(3)), float(s.group(4)), "master StationXML"))
    n = pd.read_csv(f"{T33}/national/Table/national_misorientation_final.csv")
    for r in n.drop_duplicates(["network", "station"]).itertuples():
        xy.setdefault((r.network, r.station), (r.lat, r.lon, 0.0, "33 national orientation table"))
    xy[("KS", "NAHA")] = (34.8167, 127.9261, 0.0, "Haenam renewal: master StationXML coordinates wrong (Namhae)")
    return xy


def stations():
    c = pd.read_csv(CANDIDATES); xy = coordinates()
    codes = set(c.sta)
    MERGED.update({x: x[:-1] for x in codes if x.endswith("T") and x[:-1] in codes})   # 2023 temporary codes (GWYBT, NAWBT, ...)
    rows = []
    for r in c.itertuples():
        if r.net == "KS" and r.sta in KS_DUPLICATES_OF_KG:
            continue
        if r.sta in EXCLUDE_CODES:                                            # GWANGYANG PORT (G-D20)
            print(f"excluded {r.net}.{r.sta}: {EXCLUDE_CODES[r.sta]}"); continue
        la, lo, el, src = xy[(r.net, r.sta)]
        role = f"temporary code of {MERGED[r.sta]}, merged (fills days the site's sensor lacks)" if r.sta in MERGED else "station"
        rows.append(dict(net=r.net, sta=r.sta, lat=la, lon=lo, elev=el, role=role, coords=src))
    t = pd.DataFrame(rows)
    t[["dist_km", "az_deg"]] = [dist_az(a, b) for a, b in zip(t.lat, t.lon)]
    t["site"] = [MERGED.get(s, s) for s in t.sta]
    return t


def availability():
    """days with a Z file per (net, sta, sensor, year), and the first and last day, within the window"""
    rows = []
    for line in open(AVAIL):
        y, net, sta, cha, n, a, b = line.split()
        rows.append(dict(net=net, sta=sta, sensor=cha[:2], year=int(y), days=int(n), first_doy=int(a), last_doy=int(b), source="MARA"))
    # GWANGYANG PORT (G-D14): KG days fetched from the KIGAM FDSN service after MARA ends (2024-10-31), counted from the local tree
    for d in glob.glob(f"{G}/data/continuous/KG_kigam_fdsn_sds/[0-9]*/KG/*/*Z.D"):
        p = d.split("/"); doys = sorted(int(f.rsplit(".", 1)[1]) for f in os.listdir(d))
        if doys:
            rows.append(dict(net="KG", sta=p[-2], sensor=p[-1][:2], year=int(p[-4]), days=len(doys), first_doy=doys[0], last_doy=doys[-1], source="KIGAM"))
    a = pd.DataFrame(rows)
    a = a.groupby(["net", "sta", "sensor", "year"]).agg(days=("days", "sum"), first_doy=("first_doy", "min"), last_doy=("last_doy", "max"),
                                                        source=("source", lambda x: "+".join(sorted(set(x))))).reset_index()
    y0, y1 = int(W0[:4]), int(W1[:4])
    a = a[(a.year >= y0) & (a.year <= y1)]
    a.loc[a.year == y0, "first_doy"] = a.loc[a.year == y0, "first_doy"].clip(lower=pd.Timestamp(W0).dayofyear)
    a = a[~((a.year == y0) & (a.last_doy < pd.Timestamp(W0).dayofyear))]
    a["kind"] = np.where(a.sensor.isin(["HH", "EL", "HG"]), "sensor", "other")     # BH/BG/SL are decimated copies; BA/HA as listed
    a.loc[a.year == y1, "last_doy"] = a.loc[a.year == y1, "last_doy"].clip(upper=pd.Timestamp(W1).dayofyear)
    return a


def response_epochs():
    """vertical-channel response epochs overlapping the window, from three sources"""
    from obspy import read_inventory
    rows = []
    def add(inv, src):
        for n in inv:
            for s in n:
                for c in s:
                    if not c.code.endswith("Z") or c.code[:2] not in ("HH", "EL", "HG"):
                        continue
                    a = str(c.start_date)[:10]; b = str(c.end_date)[:10] if c.end_date else "open"
                    if b != "open" and b < W0 or a > W1:
                        continue
                    sens = c.response.instrument_sensitivity.value if c.response and c.response.instrument_sensitivity else np.nan
                    rows.append(dict(net=n.code, sta=s.code, sensor=c.code[:2], start=a, end=b, sensitivity=sens, source=src))
    add(read_inventory(f"{RESP}/master/KS_KG_metadata_1.0.2.xml"), "master XML")
    for f in glob.glob(f"{RESP}/necis_fetched_2026*/extracted/RESP.*Z") + glob.glob(f"{G}/data/stations/responses/necis_fetched_*/extracted/RESP.*Z"):
        add(read_inventory(f), "NECIS current")
    for f in sorted(glob.glob(f"{G}/data/stations/responses/master/KIGAM_KG_*.xml"))[-1:]:   # GWANGYANG PORT: the latest KIGAM file of this project (all 8 KG stations)
        add(read_inventory(f), "KIGAM")
    return pd.DataFrame(rows)


def orientation():
    n = pd.read_csv(f"{T33}/national/Table/national_misorientation_final.csv")
    n = n.rename(columns={"station": "sta", "sensor": "sensor", "network": "net"})
    o = n[["net", "sta", "sensor", "start", "end", "n", "mad", "theta_M_apply"]].copy(); o["source"] = "33 national"
    # R1c-style measurements of this project (G2): 33's method on each measured unit's full history; they replace the 33 rows
    r1c = f"{G}/data/stations/orientation_r1c/Table/periods_r1c.csv"
    if os.path.exists(r1c):
        R = pd.read_csv(r1c)
        R = R[(R.method == "P_minT") & R.valid]
        units = set(zip(R.station, R.sensor))
        o = o[[(a, b) not in units for a, b in zip(o.sta, o.sensor)]]
        R2 = pd.DataFrame(dict(net=R.network, sta=R.station, sensor=R.sensor, start=R.start, end=R.end, n=R.n, mad=R.mad,
                               theta_M_apply=R.theta_M, source="G2: 33 method, full history"))
        o = pd.concat([o, R2], ignore_index=True)
    o = o[(o.end >= W0) & (o.start <= W1)]
    flags = defaultdict(list)
    fv = pd.read_csv(f"{T33}/national/Table/flip_twin_verdicts.csv")
    for r in fv.itertuples():
        if str(r.end) >= W0:
            flags[r.unit].append(f"wiring fault ({r.kind}, {r.extent}, {r.start}..{r.end}): reflect about {r.axis_pol}")
    for f, lab in [("gain_interval_corrections.csv", "horizontal gain fault"), ("turned_gain_corrections.csv", "turned-axis gain fault")]:
        g = pd.read_csv(f"{T33}/national/Table/{f}")
        for r in g.itertuples():
            if str(r.end) >= W0 and "untouched" not in str(getattr(r, "verdict", "")):
                flags[r.unit].append(f"{lab} {r.start}..{r.end}")
    s = pd.read_csv(f"{T33}/national/Table/national_silent_channel_intervals.csv")
    for r in s.itertuples():
        if str(r.end)[:10] >= W0:
            flags[r.unit].append(f"silent channel {str(r.start)[:10]}..{str(r.end)[:10]}")
    return o.sort_values(["sta", "sensor", "start"]), flags


def date_by_epochs(o, rsp):
    """Haenam rule #25, verbatim: a change bracket that holds exactly one instrument epoch of that sensor (starts within 2 days
    merged) is dated at that epoch; brackets with no epoch, or several, stay open."""
    o = o.sort_values(["sta", "sensor", "start"]).reset_index(drop=True)
    n = 0
    for (sta, sen), g in o.groupby(["sta", "sensor"]):
        e = rsp[(rsp.sta == sta) & (rsp.sensor == sen)]
        for i0, i1 in zip(list(g.index)[:-1], list(g.index)[1:]):
            a, b = pd.Timestamp(o.loc[i0, "end"]), pd.Timestamp(o.loc[i1, "start"])
            if (b - a).days <= 1:
                continue
            inside = sorted({x for x in e.start if a < pd.Timestamp(x) <= b})
            m = []
            for x in inside:
                if not m or (pd.Timestamp(x) - pd.Timestamp(m[-1])).days > 2:
                    m.append(x)
            if len(m) == 1:
                src = "/".join(sorted(set(e[e.start.isin(inside)].source)))
                o.loc[i1, "start"] = m[0]; o.loc[i0, "end"] = str((pd.Timestamp(m[0]) - pd.Timedelta(days=1)).date())
                o.loc[i1, "source"] = f"{o.loc[i1, 'source']}; change dated at the {src} epoch {m[0]}"
                n += 1
    print(f"orientation changes dated by a metadata epoch inside their bracket: {n}")
    return o


def discontinuities(sta, sen, sp, rsp, ori, oflags):
    """Haenam renewal, verbatim: (score, detail) for one sensor over its data days: orientation changes + instrument changes
    (response epochs starting inside its data period, starts within 10 days merged) + fault intervals + coverage ends short of
    the station's longest-running sensor (> 60 d). None = orientation never measured."""
    row = sp[(sp.sta == sta) & (sp.sensor == sen)]
    if row.empty:
        return None, "no data"
    d0, d1 = row.first_day.iloc[0], row.last_day.iloc[0]
    oo = ori[(ori.sta == sta) & (ori.sensor == sen) & (ori.end >= d0) & (ori.start <= d1)]
    whole = [x for x in oflags.get(f"{sta}.{sen}", []) if "whole record" in x]
    if oo.empty and not whole:
        return None, "orientation never measured"
    n_or = (len(oo) - 1) if len(oo) else 0
    starts = sorted(set(rsp[(rsp.sta == sta) & (rsp.sensor == sen) & (rsp.start > d0) & (rsp.start <= d1)].start))
    merged = []
    for x in starts:
        if not merged or (pd.Timestamp(x) - pd.Timestamp(merged[-1])).days > 10:
            merged.append(x)
    faults = [x for x in oflags.get(f"{sta}.{sen}", []) if "whole record" not in x]
    allp = sp[sp.sta == sta]
    s0, s1 = allp.first_day.min(), allp.last_day.max()
    cover = int((pd.Timestamp(d0) - pd.Timestamp(s0)).days > 60) + int((pd.Timestamp(s1) - pd.Timestamp(d1)).days > 60)
    det = (f"{n_or} orientation change(s), {len(merged)} instrument change(s) {merged}, {len(faults)} fault interval(s), "
           f"{cover} coverage end(s) short of the station's period")
    return n_or + len(merged) + len(faults) + cover, det


def main():
    st = stations(); av = availability(); rsp = response_epochs(); ori, oflags = orientation(); ori = date_by_epochs(ori, rsp)
    years = range(int(W0[:4]), int(W1[:4]) + 1)
    rows = []
    for r in st.itertuples():
        a = av[(av.net == r.net) & (av.sta == r.sta)]
        for sen in sorted(set(a.sensor)):
            g = a[a.sensor == sen].sort_values("year")
            per_year = {y: int(g[g.year == y].days.sum()) for y in years}
            if sum(per_year.values()) == 0:
                continue
            first = g.iloc[0]; last = g.iloc[-1]
            d0 = pd.Timestamp(f"{first.year}-01-01") + pd.Timedelta(days=int(first.first_doy) - 1)
            d1 = pd.Timestamp(f"{last.year}-01-01") + pd.Timedelta(days=int(last.last_doy) - 1)
            rr = rsp[(rsp.sta == r.sta) & (rsp.sensor == sen)]
            rtxt = "; ".join(f"{x.source} {x.start}..{x.end} {x.sensitivity:+.4g}" for x in rr.itertuples())
            rflag = []
            if rr.empty: rflag.append("NO RESPONSE in any source")
            elif rr.start.min() > str(d0.date()):
                lag = (pd.Timestamp(rr.start.min()) - d0).days
                rflag.append(f"note: response epoch starts {lag} d after the first data day (date conventions differ)" if lag <= 10
                             else f"NO RESPONSE before {rr.start.min()} (data from {d0.date()})")
            if (rr.sensitivity < 0).any() and (rr.sensitivity > 0).any(): rflag.append("sensitivity sign differs between epochs/sources")
            oo = ori[(ori.sta == r.sta) & (ori.sensor == sen)]
            of = oflags.get(f"{r.sta}.{sen}", [])
            whole = [x for x in of if "whole record" in x]
            stable = len(oo) == 1 and not [x for x in of if x not in whole]
            if oo.empty and whole:
                stable = True
            otxt = "; ".join(f"{x.start}..{x.end} {x.theta_M_apply:+.1f}" for x in oo.itertuples())
            rows.append(dict(net=r.net, sta=r.sta, site=r.site, role=r.role, dist_km=r.dist_km, sensor=sen,
                             **{f"days_{y}": per_year[y] for y in years}, first_day=str(d0.date()), last_day=str(d1.date()),
                             in_2021_seq=str(d0.date()) <= SEQ["2021"][1] and str(d1.date()) >= SEQ["2021"][0],
                             in_2026_seq=str(d0.date()) <= SEQ["2026"][1] and str(d1.date()) >= SEQ["2026"][0],
                             response_epochs=rtxt, response_flags="; ".join(rflag),
                             orientation_periods=otxt if otxt else ("constant wiring fault, no rotation measured" if whole else "not measured"),
                             orientation_flags="; ".join(of),
                             orientation_stable_in_window=stable if (otxt or whole) else None))
    sp = pd.DataFrame(rows)
    out = []
    for r in st.itertuples():
        g = sp[(sp.net == r.net) & (sp.sta == r.sta)]
        if g.empty:
            continue
        sensors = sorted(g.sensor)
        other = [x for x in sensors if x not in ("HH", "EL", "HG")]
        sensors = [x for x in sensors if x in ("HH", "EL", "HG")]
        if not sensors:
            continue
        days = {x.sensor: sum(getattr(x, f"days_{y}") for y in years) for x in g.itertuples() if x.sensor in sensors}
        main = [x for x in sensors if days[x] >= 60] or sensors
        stray = [f"{x} ({days[x]} d)" for x in sensors if x not in main]
        default = "HH" if "HH" in main else ("EL" if "EL" in main else ("HG" if "HG" in main else ""))
        # GWANGYANG PORT (decision G-D1b): coverage first. Over a 14-year window a sensor that starts years after the station's
        # first data (e.g. KS accelerometers from 2016-2017 beside broadbands from 2012) would cost those years if chosen; the
        # Haenam rule counts that as one discontinuity only (its window had no such case). A sensor whose data period is more
        # than COVER_SLACK_D days shorter than the station's longest-running sensor is not eligible; the discontinuity rule
        # decides among the rest.
        span = {x: (pd.Timestamp(g[g.sensor == x].last_day.iloc[0]) - pd.Timestamp(g[g.sensor == x].first_day.iloc[0])).days for x in main}
        eligible = [x for x in main if span[x] >= max(span.values()) - COVER_SLACK_D]
        dropped_cov = [x for x in main if x not in eligible]
        main = eligible
        if len(main) > 1:
            sc = {x: discontinuities(r.sta, x, sp, rsp, ori, oflags) for x in main}
            known = {x: v[0] for x, v in sc.items() if v[0] is not None}
            detail = "; ".join(f"{x}: {v[1]}" + (f" = {v[0]}" if v[0] is not None else "") for x, v in sc.items())
            if len(known) == len(main):
                best = min(known.values()); ch = next(x for x in ("HH", "EL", "HG") if known.get(x) == best)
                status = "by the sensor rule"
            else:
                ch, status = default, "current rule (a sensor's orientation never measured: rule not applicable)"
        else:
            default = main[0] if len(main) == 1 else default
            ch, detail, status = default, "", ("only sensor" if not stray else "only full-period sensor")
        if dropped_cov:
            status += f" (coverage first: {'/'.join(dropped_cov)} not eligible, data period > {COVER_SLACK_D} d shorter)"
        c = g[g.sensor == ch].iloc[0]
        faults = [x for x in (c.orientation_flags.split("; ") if isinstance(c.orientation_flags, str) else [])
                  if x.startswith("wiring fault") and "whole record" not in x]
        tm_rule = ("N/E, no rotation (orientation not measured: 0 deg)" if c.orientation_periods == "not measured" else
                   "N/E, no rotation (constant wiring fault)" if c.orientation_periods.startswith("constant wiring") else
                   "N/E" if ";" not in c.orientation_periods else "N/E, templates rotated (orientation changes)")
        if faults:
            tm_rule += "; template repair: " + "; ".join(faults)
        out.append(dict(net=r.net, sta=r.sta, site=r.site, role=r.role, lat=r.lat, lon=r.lon, coords=r.coords, dist_km=r.dist_km, az_deg=r.az_deg,
                        sensors_with_data="/".join(main), chosen_sensor=ch, choice_status=status, rule_detail=detail,
                        first_day=c.first_day, last_day=c.last_day, in_2021_seq=bool(c.in_2021_seq), in_2026_seq=bool(c.in_2026_seq),
                        s_component_rule=tm_rule, short_streams_ignored="; ".join(stray), other_channel_codes="/".join(other),
                        response_flags=c.response_flags, orientation_periods=c.orientation_periods, orientation_flags=c.orientation_flags))
    so = pd.DataFrame(out).sort_values("dist_km")
    os.makedirs(f"{HERE}/tables", exist_ok=True)
    so.to_csv(f"{HERE}/tables/stations.csv", index=False)
    sp.sort_values(["dist_km", "sta", "sensor"]).to_csv(f"{HERE}/tables/stream_periods.csv", index=False)
    ori.to_csv(f"{HERE}/tables/orientation_periods.csv", index=False)
    rsp.to_csv(f"{HERE}/tables/response_epochs.csv", index=False)
    print(f"stations {len(so)} ({so.net.value_counts().to_dict()}); station-sensor rows {len(sp)}")
    print("chosen sensors:", so.chosen_sensor.value_counts().to_dict())
    print("choice status:", so.choice_status.value_counts().to_dict())
    print("in the 2021 sequence:", int(so.in_2021_seq.sum()), "| in the 2026 sequence:", int(so.in_2026_seq.sum()))
    print("orientation not measured on the chosen sensor:", int((so.orientation_periods == "not measured").sum()))
    print("response flags on the chosen sensor:", int((so.response_flags != "").sum()))
    return so, sp


if __name__ == "__main__":
    main()
