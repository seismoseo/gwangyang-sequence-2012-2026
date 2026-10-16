#!/usr/bin/env python
"""Station inventory of the Haenam renewal: every station within 100 km with data and metadata in 2020-01-01 .. 2026-09-30.

One row per station code (tables/stations.csv) and one per station and sensor (tables/stream_periods.csv), plus an
availability figure (figures/availability.png). Read-only on every source; writes only into this folder.

Station set (user decisions 2026-10-11):
  - the 66 stations of detection/station_table/stations.csv (53 KS, 6 KG, 7 JU);
  - added for the periods they recorded: PMNM, YOA, AMD, HUK2, BOSB (decision A);
  - temporary codes merged into their sites: JAEBT -> JAEB, JEDAT -> JEDA, DAYAT -> DAYA (Jan-Jun 2023; decision A);
  - co-located second instruments, both used everywhere: KWJU (with KWJ2), NAHA (with NAJA), GOCBT (with GOCB) (decision B).
  Stations whose data or metadata we cannot get are left out (user): YGN, the 2020-2023 P-coded network except PMNM, and
  the 2023 T-codes without coordinates.

Sources:
  data      MARA SDS listing (tables/mara_availability_raw.txt, made 2026-10-11 by a read-only ssh listing);
            the KIGAM fetch log (data/continuous/KG_kigam_fdsn_sds/_manifest/fetch_log.csv; KG after 2024-10-31);
            JU day files (data/jnu/mseed/JU)
  response  master StationXML (data/stations/responses/master/KS_KG_metadata_1.0.2.xml), NECIS current-epoch RESP
            (data/stations/responses/necis_fetched_20261011), KIGAM StationXML (data/stations/responses/master/KIGAM_KG_*.xml)
  orientation  33.KS_station_orientation/national/Table/national_misorientation_final.csv (KS/KG) and Table/JU_misorientation_final.csv,
            patched where the NECIS epochs date a change to the day (PATCHES below), with the 33 repair tables as flags
Choices: SENSOR_DECIDED are the user's; SENSOR_BY_RULE come from the user's sensor rule once R1c measured them; everything else keeps
its current sensor (broadband, else short-period, else accelerometer).
"""
import glob, os, re, sys, warnings
from collections import defaultdict
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

L = "/home/msseo/works/27.Haenam_sequence"
HERE = os.path.dirname(os.path.abspath(__file__))
T33 = "/home/msseo/works/33.KS_station_orientation"
CLUSTER = (34.660, 126.405)
W0, W1 = "2020-01-01", "2026-09-30"
SEQ = {"2020": ("2020-04-26", "2020-05-31"), "2026": ("2026-08-14", "2026-09-30")}
ADDED = ["PMNM", "YOA", "AMD", "HUK2", "BOSB"]
# KMA mobile stations of the 2020 response within 15 km (KMA 2020 earthquake bulletin, table 4.4.4), added 2026-10-12 under the
# user's rule "every station within 100 km whose data we can get"; data on MARA 2020-05 .. 2023-03 (PMSM to 2021-02)
ADDED_MOBILE = ["MHAW", "PMAM", "PMSM"]
# SEJF (Seongjeon, 31 km): YOA's successor from 2024-12-23, 1.4 km from YOA (KMA bulletins 2024 and 2025; added 2026-10-12)
ADDED_LATE = ["SEJF"]
# coordinates from that bulletin (lat, lon, elevation m); for PMNM they replace the master file's two-decimal values (user, 2026-10-12)
BULLETIN = {"PMNM": (34.5974, 126.314, 65.0), "MHAW": (34.6708, 126.3277, 41.0), "PMAM": (34.754, 126.4714, 62.0), "PMSM": (34.622, 126.5679, 45.0),
            "SEJF": (34.7571, 126.7269, 202.0)}          # SEJF: KMA 2024 earthquake bulletin, station list
# a temporary code with no coordinates in any source, placed at its site by P arrival times of 14 KMA events of 2023
# (the 4 known temporary codes as controls: 2-8 km; docs/decisions/station_inventory_20261011/tables/unlocated_codes.csv)
COORD_FROM_SITE = {"DASBT": "DASB"}
MERGED = {"JAEBT": "JAEB", "JEDAT": "JEDA", "DAYAT": "DAYA", "DASBT": "DASB"}
COLOCATED = {"KWJU": "KWJ2", "NAHA": "NAJA", "GOCBT": "GOCB"}
SENSOR_DECIDED = {"HANB": "HG", "MOPB": "HG", "BGDB": "HG", "JODB": "HG",                 # user, 2026-10-11
                  "MUN": "HG", "IJDB": "HH", "BOSB": "HG",                                # user, 2026-10-11
                  "GOCB": "HH",            # user, 2026-10-12: the accelerometer has no data 2022-09..2023-07; HH turns are epoch-dated
                  # user rule 2026-10-11: short-period unless the accelerometer is steadier in orientation
                  "AJD": "EL", "JDO2": "EL", "WAN2": "EL"}                                 # both steady, or both turned together
# your sensor rule applied once R1c measured both sensors (2026-10-12; data/stations/orientation_r1c/Table/periods_r1c.csv,
# P_minT): both steady over the whole record, so the tie goes to the short-period sensor. rule_vs_choice re-checks it.
SENSOR_BY_RULE = {"YOA": ("EL", "by your rule: both steady 2020-2024 (short-period +7.6, accelerometer +2.5), tie -> short-period"),
                  "HUK2": ("EL", "by your rule: both steady 2020-2022-07 (short-period +5.2, accelerometer +4.1), tie -> short-period"),
                  "PMNM": ("EL", "by your rule: both steady 2020-05..2023-03 (short-period -6.9, accelerometer -3.8), tie -> short-period"),
                  # added 2026-10-12, measured by R1c (groups mobile, late) the same day
                  "MHAW": ("HH", "by your rule: both steady 2020-05..2023-03 (broadband -4.1, accelerometer -2.8), tie -> broadband"),
                  "PMAM": ("EL", "by your rule: both steady 2020-05..2023-03 (short-period -10.6, accelerometer -3.2), tie -> short-period"),
                  "PMSM": ("EL", "by your rule: both steady 2020-05..2021-02 (short-period -10.8, accelerometer -6.3), tie -> short-period"),
                  "SEJF": ("EL", "by your rule: both steady 2025-01..2026-06 (short-period +0.8, accelerometer +2.9), tie -> short-period"),
}
# Short-period sensors without a teleseismic value: the local-earthquake P test (docs/decisions/stream_choice_20261011,
# tables/local_vs_tele_gated.csv) carries a fixed structural offset, so it gives changes, not angles. Used only before R1c;
# R1c measured both teleseismically (JDO2.EL turned -19 -> -3 deg between 2025-03-21 and 03-25; WAN2.EL -0.1 steady).
LOCAL_P_ONLY = {("JDO2", "EL"): ("changed", "local P: -21 deg (2020) -> -7 deg (2026), re-installed 2025-01 like its HG"),
                ("WAN2", "EL"): ("steady", "local P: +3 deg (2020), +4 deg (2026)")}
# Template matching (user, 2026-10-12; replaces the 2026-10-11 S-component rule): P on Z and S on N/E at every station; where a
# sensor's orientation changes, each template's N/E are turned into the orientation of the day being scanned (data never rotated).
# NECIS epochs that date an orientation change to the day (necis_fetched_20261011/compare_with_master.csv)
PATCHES = [("HANB", "HG", "2026-08-12", -173.40, "synthesis/Table/KS_JU_misorientation_final.csv L12 (n=7); local P; NECIS epoch 2026-08-12"),
           ("HANB", "HH", "2026-04-15", -32.22, "national L533; start pinned to the NECIS epoch 2026-04-15"),
           ("JODB", "HH", "2026-08-11", 30.65, "national L703; start pinned to the NECIS epoch 2026-08-11")]
# whole-record wiring faults that 33 dated only to its data end; R1c's later events still lie on the same reflection line
FAULT_EXTENDED = {"YSDA.HG": (W1, " (continues: R1c events 2026-09-11 and 09-25 on the same reflection line)")}


def dist_az(lat, lon):
    from obspy.geodetics import gps2dist_azimuth
    d, az, _ = gps2dist_azimuth(CLUSTER[0], CLUSTER[1], lat, lon)
    return round(d / 1e3, 1), round(az)


def stations():
    from obspy import read_inventory
    dl = pd.read_csv(f"{L}/detection/station_table/stations.csv")
    rows = [dict(net=r.Network, sta=r.Code, lat=r.Latitude, lon=r.Longitude, elev=r.Elevation, role="detection table",
                 coords="detection station table") for r in dl.itertuples()]
    inv = read_inventory(f"{L}/data/stations/responses/master/KS_KG_metadata_1.0.2.xml")
    xy = {s.code: (n.code, s.latitude, s.longitude, s.elevation) for n in inv for s in n}
    site = {r.Code: (r.Network, r.Latitude, r.Longitude, r.Elevation) for r in dl.itertuples()}
    for s in ADDED + ADDED_MOBILE + ADDED_LATE + list(MERGED) + list(COLOCATED):
        n, la, lo, el = ("KS", *BULLETIN[s]) if s in BULLETIN else site[COORD_FROM_SITE[s]] if s in COORD_FROM_SITE else xy[s]
        role = ("added (decision A)" if s in ADDED else "added: KMA mobile station of the 2020 response" if s in ADDED_MOBILE else
                "added: YOA's successor from 2024-12-23" if s in ADDED_LATE else
                f"temporary code of {MERGED[s]}, merged ({'as decision A, 2026-10-12' if s == 'DASBT' else 'decision A'})" if s in MERGED else
                f"co-located with {COLOCATED[s]}, both used (decision B)")
        rows.append(dict(net=n, sta=s, lat=la, lon=lo, elev=el, role=role,
                         coords=("KMA 2024 earthquake bulletin, station list" if s == "SEJF" else "KMA 2020 earthquake bulletin, table 4.4.4") if s in BULLETIN else
                                f"the {COORD_FROM_SITE[s]} site (P arrival times, 2026-10-12)" if s in COORD_FROM_SITE else "master StationXML"))
    t = pd.DataFrame(rows)
    t[["dist_km", "az_deg"]] = [dist_az(a, b) for a, b in zip(t.lat, t.lon)]
    t["site"] = [MERGED.get(s, s) for s in t.sta]
    return t


def availability():
    """days with a Z file per (net, sta, sensor, year), and the dates of the first and last day, within the window"""
    rows = []
    for line in open(f"{HERE}/tables/mara_availability_raw.txt"):
        y, net, sta, cha, n, a, b = line.split()
        rows.append(dict(net=net, sta=sta, sensor=cha[:2], year=int(y), days=int(n), first_doy=int(a), last_doy=int(b), source="MARA"))
    fl = f"{L}/data/continuous/KG_kigam_fdsn_sds/_manifest/fetch_log.csv"
    if os.path.exists(fl):
        k = pd.read_csv(fl, dtype=str)
        k = k[(k.status == "ok") & k.channel.str.endswith("Z")]
        k["year"] = k.day.str[:4].astype(int); k["doy"] = pd.to_datetime(k.day).dt.dayofyear
        for (sta, cha, y), g in k.groupby(["station", "channel", "year"]):
            rows.append(dict(net="KG", sta=sta, sensor=cha[:2], year=y, days=len(g), first_doy=g.doy.min(), last_doy=g.doy.max(), source="KIGAM"))
    for f in glob.glob(f"{L}/data/jnu/mseed/JU/*/*.HHZ.*"):
        m = re.search(r"/([A-Z0-9]+)\.JU\.\d*\.HHZ\.(\d{4})\.(\d{3})$", f)
        if m:
            rows.append(dict(net="JU", sta=m.group(1), sensor="HH", year=int(m.group(2)), days=1, first_doy=int(m.group(3)),
                             last_doy=int(m.group(3)), source="JU files"))
    a = pd.DataFrame(rows)
    a = a.groupby(["net", "sta", "sensor", "year", "source"]).agg(days=("days", "sum"), first_doy=("first_doy", "min"),
                                                                    last_doy=("last_doy", "max")).reset_index()
    a = a[(a.year >= 2020) & (a.year <= 2026)]
    a["kind"] = np.where(a.sensor.isin(["HH", "EL", "HG"]), "sensor", "other")     # BH/BG/SL are decimated copies; BA/HA as listed
    a.loc[a.year == 2026, "last_doy"] = a.loc[a.year == 2026, "last_doy"].clip(upper=273)
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
    add(read_inventory(f"{L}/data/stations/responses/master/KS_KG_metadata_1.0.2.xml"), "master XML")
    for f in glob.glob(f"{L}/data/stations/responses/necis_fetched_2026*/extracted/RESP.*Z"):
        add(read_inventory(f), "NECIS current")
    for f in sorted(glob.glob(f"{L}/data/stations/responses/master/KIGAM_KG_*.xml"))[-1:]:
        add(read_inventory(f), "KIGAM")
    add(read_inventory(f"{L}/data/jnu/JU_station.xml"), "JU StationXML")
    return pd.DataFrame(rows)


def orientation():
    n = pd.read_csv(f"{T33}/national/Table/national_misorientation_final.csv")
    n = n.rename(columns={"station": "sta", "sensor": "sensor", "network": "net"})
    o = n[["net", "sta", "sensor", "start", "end", "n", "mad", "theta_M_apply"]].copy(); o["source"] = "33 national"
    j = pd.read_csv(f"{T33}/Table/JU_misorientation_final.csv")           # station,method,start,end,n,median,mad,...
    j = j[(j.method == "P_minT") & j.valid]                                  # the method the national table uses
    for r in j.itertuples():
        o.loc[len(o)] = dict(net="JU", sta=r.station, sensor="HH", start=r.start, end=r.end,
                             n=r.n, mad=r.mad, theta_M_apply=r.median, source="33 JU (P_minT)")
    # R1c (2026-10-11): 33's method re-run on each measured unit's FULL history (KG after 2024-10, the added stations,
    # every station extended to 2026-09-30); for those units it replaces the national/JU rows
    r1c = f"{L}/data/stations/orientation_r1c/Table/periods_r1c.csv"
    r1c_units = set()
    if os.path.exists(r1c):
        R = pd.read_csv(r1c)
        R = R[(R.method == "P_minT") & R.valid]
        r1c_units = set(zip(R.station, R.sensor))
        o = o[[(a, b) not in r1c_units for a, b in zip(o.sta, o.sensor)]]
        R2 = pd.DataFrame(dict(net=R.network, sta=R.station, sensor=R.sensor, start=R.start, end=R.end, n=R.n, mad=R.mad,
                               theta_M_apply=R.theta_M, source="R1c: 33 method, full history"))
        o = pd.concat([o, R2], ignore_index=True)
    for sta, sen, day, th, src in PATCHES:
        if (sta, sen) in r1c_units:                    # R1c shows this change: keep its angle, date it at the epoch
            g = o[(o.sta == sta) & (o.sensor == sen)].sort_values("start")
            hit = [i for i, x, y in zip(g.index, g.start, g.theta_M_apply)
                   if abs((pd.Timestamp(x) - pd.Timestamp(day)).days) <= 15 and abs(((y - th) + 180) % 360 - 180) <= 15]
            if hit:
                i = hit[0]; before = g[g.start < g.loc[i, "start"]]
                o.loc[i, "start"] = day; o.loc[i, "source"] += f"; change dated at the NECIS epoch {day}"
                if len(before):                        # the period before ends the day before the epoch
                    o.loc[before.index[-1], "end"] = str((pd.Timestamp(day) - pd.Timedelta(days=1)).date())
                continue
        g = o[(o.sta == sta) & (o.sensor == sen)]
        prev = g[(g.start <= day) & (g.end >= day)]
        for i in prev.index:                                          # close the period that runs across the change
            o.loc[i, "end"] = str((pd.Timestamp(day) - pd.Timedelta(days=1)).date())
        late = g[(g.start > day) & (np.isclose(g.theta_M_apply, th))]
        o = o.drop(late.index)
        o = pd.concat([o, pd.DataFrame([dict(net="KS", sta=sta, sensor=sen, start=day, end=W1, n=np.nan, mad=np.nan,
                                             theta_M_apply=th, source=f"patch: {src}")])], ignore_index=True)
    o = o[(o.end >= W0) & (o.start <= W1)]
    flags = defaultdict(list)
    fv = pd.read_csv(f"{T33}/national/Table/flip_twin_verdicts.csv")
    for r in fv.itertuples():
        if str(r.end) >= W0:
            end, why = FAULT_EXTENDED.get(r.unit, (r.end, "")) if r.extent == "whole record" else (r.end, "")
            flags[r.unit].append(f"wiring fault ({r.kind}, {r.extent}, {r.start}..{end}): reflect about {r.axis_pol}{why}")
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
    """A change bracket (days between the last measurement before a change and the first after it) that holds exactly one
    instrument epoch of that sensor, in any metadata source (starts within 2 days merged), is dated at that epoch: the new
    period starts that day. Brackets with no epoch, or several, stay open (ISSUES #25)."""
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
    """(score, detail) for one sensor in the window, over the days it has data: orientation changes + instrument changes
    (response epochs starting inside its data period, starts within 10 days merged) + known fault intervals. None = unknown."""
    row = sp[(sp.sta == sta) & (sp.sensor == sen)]
    if row.empty:
        return None, "no data"
    d0, d1 = row.first_day.iloc[0], row.last_day.iloc[0]
    oo = ori[(ori.sta == sta) & (ori.sensor == sen) & (ori.end >= d0) & (ori.start <= d1)]
    whole = [x for x in oflags.get(f"{sta}.{sen}", []) if "whole record" in x]
    local = (sta, sen) in LOCAL_P_ONLY
    if oo.empty and not whole and not local:
        return None, "orientation never measured"
    n_or = (len(oo) - 1) if len(oo) else (1 if local and LOCAL_P_ONLY[(sta, sen)][0] == "changed" else 0)
    starts = sorted(set(rsp[(rsp.sta == sta) & (rsp.sensor == sen) & (rsp.start > d0) & (rsp.start <= d1)].start))
    merged = []
    for x in starts:
        if not merged or (pd.Timestamp(x) - pd.Timestamp(merged[-1])).days > 10:
            merged.append(x)
    faults = [x for x in oflags.get(f"{sta}.{sen}", []) if "whole record" not in x and not x.startswith("local P")]
    # a sensor that starts later or ends earlier than the station's longest-running sensor (by > 60 d) forces a switch there
    allp = sp[sp.sta == sta]
    s0, s1 = allp.first_day.min(), allp.last_day.max()
    cover = int((pd.Timestamp(d0) - pd.Timestamp(s0)).days > 60) + int((pd.Timestamp(s1) - pd.Timestamp(d1)).days > 60)
    det = (f"{n_or} orientation change(s), {len(merged)} instrument change(s) {merged}, {len(faults)} fault interval(s), "
           f"{cover} coverage end(s) short of the station's period")
    return n_or + len(merged) + len(faults) + cover, det


def apply_sensor_rule(so, sp, rsp, ori, oflags):
    """the user's sensor rule (2026-10-11), applied to every station with more than one sensor"""
    rule, detail, agree = [], [], []
    for r in so.itertuples():
        sensors = [x for x in r.sensors_with_data.split("/") if x]
        if len(sensors) < 2:
            rule.append(r.chosen_sensor); detail.append(""); agree.append(""); continue
        sc = {x: discontinuities(r.sta, x, sp, rsp, ori, oflags) for x in sensors}
        txt = "; ".join(f"{x}: {v[1]}" + (f" = {v[0]}" if v[0] is not None else "") for x, v in sc.items())
        known = {x: v[0] for x, v in sc.items() if v[0] is not None}
        if len(known) < len(sensors):
            pick = "unknown"
        else:
            best = min(known.values())
            pick = next(x for x in ("HH", "EL", "HG") if known.get(x) == best)
        rule.append(pick); detail.append(txt)
        agree.append("" if pick == "unknown" else ("same" if pick == r.chosen_sensor else f"DIFFERS from the chosen {r.chosen_sensor}"))
    so = so.copy(); so["rule_sensor"] = rule; so["rule_detail"] = detail; so["rule_vs_choice"] = agree
    return so


def main():
    st = stations(); av = availability(); rsp = response_epochs(); ori, oflags = orientation(); ori = date_by_epochs(ori, rsp)
    rows = []
    for r in st.itertuples():
        a = av[(av.sta == r.sta)]
        for sen in sorted(set(a.sensor)):
            g = a[a.sensor == sen].sort_values("year")
            per_year = {y: int(g[g.year == y].days.sum()) for y in range(2020, 2027)}
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
            whole = [x for x in of if "whole record" in x]                      # a constant wiring fault: raw N/E stay consistent
            stable = len(oo) == 1 and not [x for x in of if x not in whole]
            if oo.empty and (r.sta, sen) in LOCAL_P_ONLY and not os.path.exists(f"{L}/data/stations/orientation_r1c/Table/periods_r1c.csv"):
                st_, note = LOCAL_P_ONLY[(r.sta, sen)]
                oo = pd.DataFrame([dict(start="", end="", theta_M_apply=np.nan)]); stable = st_ == "steady"; of = of + [note]
            elif oo.empty and whole:
                stable = True
            otxt = "; ".join(f"{x.start}..{x.end} {x.theta_M_apply:+.1f}" for x in oo.itertuples())
            rows.append(dict(net=r.net, sta=r.sta, site=r.site, role=r.role, dist_km=r.dist_km, sensor=sen,
                             **{f"days_{y}": per_year[y] for y in range(2020, 2027)}, first_day=str(d0.date()), last_day=str(d1.date()),
                             in_2020_seq=str(d0.date()) <= SEQ["2020"][1] and str(d1.date()) >= SEQ["2020"][0],
                             in_2026_seq=str(d0.date()) <= SEQ["2026"][1] and str(d1.date()) >= SEQ["2026"][0],
                             response_epochs=rtxt, response_flags="; ".join(rflag),
                             orientation_periods=otxt if otxt else ("constant wiring fault, no rotation measured" if whole else
                                                                     "local P only" if (r.sta, sen) in LOCAL_P_ONLY else "not measured"),
                             orientation_flags="; ".join(of),
                             orientation_stable_in_window=stable if (otxt or whole or (r.sta, sen) in LOCAL_P_ONLY) else None))
    sp = pd.DataFrame(rows)
    out = []
    for r in st.itertuples():
        g = sp[sp.sta == r.sta]
        sensors = sorted(g.sensor)
        other = [x for x in sensors if x not in ("HH", "EL", "HG")]
        sensors = [x for x in sensors if x in ("HH", "EL", "HG")]
        days = {x.sensor: sum(getattr(x, f"days_{y}") for y in range(2020, 2027)) for x in g.itertuples() if x.sensor in sensors}
        main = [x for x in sensors if days[x] >= 60] or sensors        # under two months is stray only beside a full-period sensor
        stray = [f"{x} ({days[x]} d)" for x in sensors if x not in main]
        default = "HH" if "HH" in main else ("EL" if "EL" in main else ("HG" if "HG" in main else ""))
        if r.sta in SENSOR_DECIDED: ch, status, why = SENSOR_DECIDED[r.sta], "decided (user)", ""
        elif r.sta in SENSOR_BY_RULE and SENSOR_BY_RULE[r.sta][0] in sensors: ch, status, why = SENSOR_BY_RULE[r.sta][0], "by your rule (R1c)", SENSOR_BY_RULE[r.sta][1]
        elif len(main) > 1: ch, status, why = default, "unchanged (current rule)", ""
        else: ch, status, why = default, "only sensor" if not stray else "only full-period sensor", ""
        c = g[g.sensor == ch].iloc[0] if ch in set(g.sensor) else None
        faults = [x for x in (c.orientation_flags.split("; ") if c is not None and isinstance(c.orientation_flags, str) else [])
                  if x.startswith("wiring fault") and "whole record" not in x]
        tm_rule = "" if c is None else (
            "N/E, no rotation (orientation not measured: 0 deg)" if c.orientation_periods == "not measured" else
            "N/E, no rotation (constant wiring fault)" if c.orientation_periods.startswith("constant wiring") else
            "N/E" if ";" not in c.orientation_periods else "N/E, templates rotated (orientation changes)")
        if faults:
            tm_rule += "; template repair: " + "; ".join(faults)
        out.append(dict(net=r.net, sta=r.sta, site=r.site, role=r.role, lat=r.lat, lon=r.lon, coords=r.coords, dist_km=r.dist_km, az_deg=r.az_deg,
                        sensors_with_data="/".join(main), chosen_sensor=ch, choice_status=status, reason=why,
                        first_day=c.first_day if c is not None else "", last_day=c.last_day if c is not None else "",
                        in_2020_seq=bool(c.in_2020_seq) if c is not None else False, in_2026_seq=bool(c.in_2026_seq) if c is not None else False,
                        s_component_rule=tm_rule, short_streams_ignored="; ".join(stray), other_channel_codes="/".join(other),
                        response_flags=c.response_flags if c is not None else "no data", orientation_flags=c.orientation_flags if c is not None else ""))
    so = pd.DataFrame(out).sort_values("dist_km")
    so = apply_sensor_rule(so, sp, rsp, ori, oflags)
    os.makedirs(f"{HERE}/tables", exist_ok=True)
    so.to_csv(f"{HERE}/tables/stations.csv", index=False)
    sp.sort_values(["dist_km", "sta", "sensor"]).to_csv(f"{HERE}/tables/stream_periods.csv", index=False)
    ori.to_csv(f"{HERE}/tables/orientation_periods.csv", index=False)
    rsp.to_csv(f"{HERE}/tables/response_epochs.csv", index=False)
    print(f"stations {len(so)} ({so.net.value_counts().to_dict()}); station-sensor rows {len(sp)}")
    print("choice status:", so.choice_status.value_counts().to_dict())
    print("in the 2020 sequence:", int(so.in_2020_seq.sum()), "| in the 2026 sequence:", int(so.in_2026_seq.sum()))
    print("response flags on the chosen sensor:", int((so.response_flags != "").sum()))
    return so, sp


if __name__ == "__main__":
    main()
