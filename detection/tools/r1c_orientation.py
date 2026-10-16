#!/usr/bin/env python
"""Renewal step R1c: the orientation measurements still missing, by the 33 study's own method.

What is measured (each 'unit' = station + sensor):
  kg      the 6 KG stations after 2024-10-31 (KIGAM fetch; 33 stopped where the KG archive ended)
  added   YOA, HUK2, PMNM, short-period and accelerometer (MARA data from step R1b), over the years they recorded
  extend  every KS station of the inventory and the JU stations, 2026-07-01 .. 2026-09-30 (33 ends 2026-08-24 / 09-03)
  mobile  MHAW, PMAM, PMSM: KMA mobile stations of the 2020 response, both sensors, 2020-05 .. 2023-03 (added 2026-10-12)
  late    SEJF (YOA's successor, 2025-01 .. 2026-06) and DASBT (DASB's 2023 temporary code) (added 2026-10-12)

How (identical to 33.KS_station_orientation/national, whose functions are imported, read-only):
  windows   as fetch_national_stp.do_chunk: theoretical first P (iasp91), 300 s around it, 20 sps, 0.02-0.2 Hz;
            only the source differs: local day files instead of the STP server (KIGAM tree, MARA mirror, NECIS
            cron archive for KS 2026, JU day files)
  measure   measure_national.measure_event (P_PCA and P_minT, the same windows, QC and sign convention)
  periods   synthesize_national.process_unit on each unit's FULL history: 33's measurements_all.csv rows plus the
            new events (33's row wins where both measured the same event; those pairs check the local cutting)
Catalog: data/jnu/teleseisms_M5.5_90deg_2016_2026.csv (to 2026-09-14) plus a USGS supplement to 2026-09-30, same
selection (M >= 5.5, <= 90 deg from 34.66 N 126.40 E).
Outputs (read-only on every source): data/stations/orientation_r1c/{cut,meas,Table}.

  python detection/tools/r1c_orientation.py catalog        # the USGS supplement
  python detection/tools/r1c_orientation.py cut --group kg|added|extend|mobile [--workers 8]
  python detection/tools/r1c_orientation.py measure [--workers 8]
  python detection/tools/r1c_orientation.py periods
"""
import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # the project registry, found upward from this file
import argparse, glob, os, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

L = haenam.ROOT
N33 = "/home/msseo/works/33.KS_station_orientation/national"
sys.path.insert(0, N33)
OUT = L / "data/stations/orientation_r1c"
CUT, MEAS, TAB = OUT / "cut", OUT / "meas", OUT / "Table"
CAT_BASE = pathlib.Path("/home/msseo/works/27.Haenam_sequence/data/jnu/teleseisms_M5.5_90deg_2016_2026.csv")   # GWANGYANG PORT: read from the Haenam tree
CAT_SUPP = TAB / "teleseisms_supplement_gwangyang.csv"          # GWANGYANG PORT: USGS 2012-12-01..2016-01-01 and 2026-09-14..2026-10-07
REF = (35.0000, 127.7286)                                       # GWANGYANG PORT: the Gwangyang centroid
INV = L / "docs/station_inventory/tables"                       # GWANGYANG PORT
SRC = {"kigam": L / "data/continuous/KG_kigam_fdsn_sds", "mara": L / "data/continuous/KS_KG_mara_sds",
       "necis": pathlib.Path("/home/msseo/works/Claude/data/necis/continuous"), "jnu": L / "data/jnu/mseed/JU"}
TARGET_SPS, FREQ_BAND = 20, (0.02, 0.2)          # = fetch_national_stp
CUT_PRE, CUT_POST, WIN_PAD = 120.0, 180.0, 10.0
MIN_MAG, MAX_DIST = 5.5, 90.0


def units(group):
    st = pd.read_csv(INV / "stations.csv")
    xy = {r.sta: (r.lat, r.lon) for r in st.itertuples()}
    u = []
    if group == "kg":
        for sta, bands in [("AJD", "EL HG"), ("MUN", "HH HG"), ("GHR", "HG"), ("BGD", "HH HG"), ("HWSB", "HH HG"), ("IBA", "HG")]:
            for b in bands.split():
                u.append(("KG", sta, b, "2024-11-01", "2026-09-30", "kigam", *xy[sta]))
    elif group == "added":
        for sta, a, e in [("YOA", "2020-01-01", "2024-12-31"), ("HUK2", "2020-01-01", "2022-07-31"), ("PMNM", "2020-05-06", "2023-03-31")]:
            for b in ("EL", "HG"):
                u.append(("KS", sta, b, a, e, "mara", *xy[sta]))
        # short-period sensors that 33 never measured (only the local-earthquake test so far): the whole window
        for sta in ("JDO2", "WAN2"):
            u.append(("KS", sta, "EL", "2020-01-01", "2026-09-30", "mara", *xy[sta]))
        # the 2023 temporary codes (January - June 2023), otherwise assumed at 0 deg
        for sta, bands in [("JAEBT", "HH HG"), ("JEDAT", "HG"), ("DAYAT", "HG"), ("GOCBT", "HH HG")]:
            for b in bands.split():
                u.append(("KS", sta, b, "2023-01-01", "2023-07-03", "mara", *xy[sta]))
    elif group == "mobile":
        # KMA mobile stations of the 2020 response (bulletin coordinates; MARA data fetched 2026-10-12)
        for sta, bands, a, e in [("MHAW", "HH HG", "2020-05-05", "2023-03-14"), ("PMAM", "EL HG", "2020-05-07", "2023-03-14"),
                                 ("PMSM", "EL HG", "2020-05-07", "2021-02-08")]:
            for b in bands.split():
                u.append(("KS", sta, b, a, e, "mara", *xy[sta]))
    elif group == "late":
        # SEJF (YOA's successor, MARA data 2025-01-07 .. 2026-06-30) and DASBT (DASB's 2023 temporary code); added 2026-10-12
        for sta, bands, a, e in [("SEJF", "EL HG", "2025-01-07", "2026-06-30"), ("DASBT", "HH HG", "2023-01-12", "2023-07-03")]:
            for b in bands.split():
                u.append(("KS", sta, b, a, e, "mara", *xy[sta]))
    elif group == "gwangyang":
        # GWANGYANG PORT: every unit of the Gwangyang inventory that needs measuring (decision G-D2):
        #  - chosen sensors that 33 never measured, and the other sensors of stations where the sensor rule is pending: whole record;
        #  - data before a unit's first 33 measurement (e.g. GWYB.HH 2012-12, KS accelerometers 2016-2017): data start .. that day;
        #  - every chosen unit with data after 2026-07-01: 2026-07-01 .. its last day (33 ends 2026-08-24 / 09-03).
        st = pd.read_csv(INV / "stations.csv"); sp = pd.read_csv(INV / "stream_periods.csv")
        n33 = pd.read_csv("/home/msseo/works/33.KS_station_orientation/national/Table/national_misorientation_final.csv")
        first33 = n33.groupby(["station", "sensor"]).start.min().to_dict()
        pending = set(st[st.choice_status.str.startswith("current rule")].sta)
        chosen = {(r.net, r.sta): r.chosen_sensor for r in st.itertuples()}
        for r in sp[sp.sensor.isin(["HH", "EL", "HG"])].itertuples():
            ch = chosen.get((r.net, r.sta))
            if ch is None or (r.sensor != ch and r.sta not in pending):
                continue
            src = "mara"
            f33 = first33.get((r.sta, r.sensor))
            if f33 is None:
                u.append((r.net, r.sta, r.sensor, r.first_day, r.last_day, src, *xy[r.sta])); continue
            if r.first_day < f33:
                u.append((r.net, r.sta, r.sensor, r.first_day, str((pd.Timestamp(f33) - pd.Timedelta(days=1)).date()), src, *xy[r.sta]))
            if r.last_day >= "2026-07-01":
                u.append((r.net, r.sta, r.sensor, "2026-07-01", r.last_day, src, *xy[r.sta]))
    elif group == "extend":
        sp = pd.read_csv(INV / "stream_periods.csv")
        for r in sp[(sp.last_day >= "2026-07-01") & sp.sensor.isin(["HH", "EL", "HG"])].itertuples():
            if r.net == "KS":
                u.append(("KS", r.sta, r.sensor, "2026-07-01", "2026-09-30", "necis", *xy[r.sta]))
            elif r.net == "JU":
                u.append(("JU", r.sta, "HH", "2025-12-01", "2026-09-30", "jnu", *xy[r.sta]))
    return u


def catalog():
    c = pd.read_csv(CAT_BASE, parse_dates=["time"])
    if CAT_SUPP.exists():
        c = pd.concat([c, pd.read_csv(CAT_SUPP, parse_dates=["time"])]).drop_duplicates("time")
    return c[(c.mag >= MIN_MAG) & (c.dist_deg <= MAX_DIST)].sort_values("time").reset_index(drop=True)


def cmd_catalog(a):
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    from obspy.geodetics import locations2degrees, gps2dist_azimuth
    ev = []                                                     # GWANGYANG PORT: the two spans the Haenam catalog does not cover
    for t0, t1 in (("2012-12-01", "2016-01-01"), ("2026-09-14T11:00:00", "2026-10-07")):
        ev += list(Client("USGS").get_events(starttime=UTCDateTime(t0), endtime=UTCDateTime(t1), minmagnitude=MIN_MAG,
                                             latitude=REF[0], longitude=REF[1], maxradius=MAX_DIST))
    rows = []
    for e in ev:
        o = e.preferred_origin() or e.origins[0]; m = e.preferred_magnitude() or e.magnitudes[0]
        rows.append(dict(time=o.time.datetime.strftime("%Y-%m-%dT%H:%M:%S"), lat=o.latitude, lon=o.longitude,
                         depth_km=(o.depth or 0) / 1e3, mag=m.mag, magtype=m.magnitude_type,
                         dist_deg=round(locations2degrees(REF[0], REF[1], o.latitude, o.longitude), 2),
                         baz=round(gps2dist_azimuth(REF[0], REF[1], o.latitude, o.longitude)[1], 1)))
    os.makedirs(TAB, exist_ok=True)
    pd.DataFrame(rows).sort_values("time").to_csv(CAT_SUPP, index=False)
    print(f"USGS supplement: {len(rows)} events M>={MIN_MAG} <= {MAX_DIST} deg (2012-12..2016 and 2026-09-14..10-06) -> {CAT_SUPP}")


def day_files(src, net, sta, cha, day):
    y, j = day.year, day.timetuple().tm_yday
    if src in ("kigam", "mara"):
        return glob.glob(str(SRC[src] / f"{y}/{net}/{sta}/{cha}.D/{net}.{sta}.*.{cha}.D.{y}.{j:03d}"))
    if src == "necis":
        return glob.glob(str(SRC[src] / f"{y}/{sta}/{net}.{sta}.{cha}.{y}.{j:03d}.*"))
    if src == "jnu":
        return glob.glob(str(SRC[src] / f"{sta}/{sta}.{net}.*.{cha}.{y}.{j:03d}"))
    return []


def cut_unit(u):
    """all events of one unit: the 33 window processing on local day files. Returns windows written."""
    import datetime as dt
    from obspy import read, UTCDateTime, Stream
    from obspy.geodetics import gps2dist_azimuth, locations2degrees
    from fetch_national_stp import first_p
    net, sta, band, a, e, src, la, lo = u
    cat = catalog(); cat = cat[(cat.time >= a) & (cat.time <= pd.Timestamp(e) + pd.Timedelta(days=1))]
    n = 0
    for ev in cat.itertuples():
        eid = ev.time.strftime("%Y%m%d%H%M%S")
        out = CUT / eid / f"{eid}.{net}.{sta}.{band}E.sac"
        if out.exists():
            continue
        dist_m, baz, az = gps2dist_azimuth(la, lo, ev.lat, ev.lon)
        gcarc = locations2degrees(la, lo, ev.lat, ev.lon); tp = first_p(gcarc, ev.depth_km)
        if tp is None:
            continue
        Tp = UTCDateTime(ev.time.to_pydatetime()) + tp
        t0, t1 = Tp - CUT_PRE - WIN_PAD, Tp + CUT_POST + WIN_PAD
        trs = []
        for c in "ZNE":
            fs = sorted({f for d in {t0.date, t1.date} for f in day_files(src, net, sta, f"{band}{c}", d)})
            if not fs:
                trs = None; break
            st = Stream()
            for f in fs:
                try: st += read(f, starttime=t0 - 1, endtime=t1 + 1)
                except Exception: pass
            for tr in st: tr.stats.sampling_rate = float(round(tr.stats.sampling_rate))
            st.merge(method=1, fill_value=None)
            if len(st) != 1 or np.ma.is_masked(st[0].data):
                trs = None; break
            tr = st[0]
            if not (tr.stats.starttime <= t0 + 1 and tr.stats.endtime >= t1 - 1):
                trs = None; break
            tr.stats.network, tr.stats.station, tr.stats.channel = net, sta, f"{band}{c}"
            trs.append(tr)
        if trs is None:
            continue
        st = Stream(trs); st.detrend("demean").detrend("linear").taper(0.05)
        fac = int(round(st[0].stats.sampling_rate / TARGET_SPS))
        if fac > 1: st.decimate(fac, no_filter=False)
        st.filter("bandpass", freqmin=FREQ_BAND[0], freqmax=FREQ_BAND[1], corners=4, zerophase=True)
        st.trim(Tp - CUT_PRE, Tp + CUT_POST)
        os.makedirs(CUT / eid, exist_ok=True)
        for tr, c in zip(st, "ZNE"):
            tr.stats.sac = dict(evla=ev.lat, evlo=ev.lon, evdp=ev.depth_km, mag=ev.mag, stla=la, stlo=lo, baz=baz, az=az,
                                gcarc=gcarc, o=-(CUT_PRE + tp), t0=CUT_PRE, kt0="Ptheo", b=0.0)
            tr.write(str(CUT / eid / f"{eid}.{net}.{sta}.{band}{c}.sac"), format="SAC")
        n += 1
    return f"{net}.{sta}.{band}", n


def cmd_cut(a):
    us = units(a.group)
    print(f"cut {a.group}: {len(us)} units, {a.workers} workers", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for name, n in ex.map(cut_unit, us):
            print(f"  {name}: {n} windows", flush=True)


def _measure(eid):
    import measure_national as mn
    mn.CUT, mn.MEAS = str(CUT), str(MEAS)
    return eid, mn.measure_event(eid)


def cmd_measure(a):
    os.makedirs(MEAS, exist_ok=True)
    eids = sorted(os.path.basename(d) for d in glob.glob(str(CUT / "*")))
    print(f"measure: {len(eids)} events, {a.workers} workers", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        n = sum(1 for _ in ex.map(_measure, eids, chunksize=8))
    fs = sorted(glob.glob(str(MEAS / "*.csv")))
    m = pd.concat([pd.read_csv(f) for f in fs if os.path.getsize(f) > 1], ignore_index=True)
    m.to_csv(TAB / "measurements_r1c.csv", index=False)
    print(f"  {len(m)} unit-event measurements -> {TAB / 'measurements_r1c.csv'}")


def cmd_periods(a):
    from synthesize_national import process_unit
    new = pd.read_csv(TAB / "measurements_r1c.csv", parse_dates=["time"])
    old = pd.read_csv(f"{N33}/Table/measurements_all.csv", parse_dates=["time"],
                      usecols=lambda c: c in set(new.columns))
    old = old[old.unit.isin(set(new.unit))]
    key = lambda d: d.unit + "|" + d.event.astype(str)
    both = new[key(new).isin(set(key(old)))].merge(old, on=["unit", "event"], suffixes=("_r1c", "_33"))
    if len(both):                                      # the same event measured from STP (33) and from local files (here)
        ok = both[both.ok_mint_r1c & both.ok_mint_33]
        d = np.abs((ok.mis_mint_r1c - ok.mis_mint_33 + 180) % 360 - 180)
        pd.DataFrame(dict(unit=ok.unit, event=ok.event, diff=d)).to_csv(TAB / "check_local_vs_stp.csv", index=False)
        print(f"check: {len(both)} unit-events measured both ways; accepted by both {len(ok)}; |diff| median {d.median():.2f} deg, "
              f"95% {d.quantile(0.95):.2f} deg, max {d.max():.2f} deg")
    full = pd.concat([old, new[~key(new).isin(set(key(old)))]], ignore_index=True)
    rows = []
    for unit, g in full.groupby("unit"):
        out, per, mir, sil = process_unit((unit, g))
        for p in per:
            p.update(mirror=out.get("mirror", ""), silent=out.get("silent", ""), n_new=int(g.event.isin(new[new.unit == unit].event).sum()))
        rows += per
    P = pd.DataFrame(rows)
    P.to_csv(TAB / "periods_r1c.csv", index=False)
    pm = P[(P.method == "P_minT")]
    print(f"periods: {pm.unit.nunique()} units, {len(pm)} P_minT periods (valid {int(pm.valid.sum())}) -> {TAB / 'periods_r1c.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("catalog")
    p = sub.add_parser("cut"); p.add_argument("--group", required=True, choices=["kg", "added", "extend", "mobile", "late", "gwangyang"]); p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("measure"); p.add_argument("--workers", type=int, default=8)
    sub.add_parser("periods")
    a = ap.parse_args()
    os.makedirs(TAB, exist_ok=True)
    return {"catalog": cmd_catalog, "cut": cmd_cut, "measure": cmd_measure, "periods": cmd_periods}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
