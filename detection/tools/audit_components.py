#!/usr/bin/env python
"""GWANGYANG (decision G-D13): component completeness of the continuous link tree, per chosen sensor and day.

For every row of the station table (one chosen sensor per code) and every day of the requested years inside the row's
Start .. min(End, LinkUntil), it lists which of Z/N/E have a day file in data/continuous/sds_links. A station-day with one
or two components missing is "partial"; with none it is "no data". Partial days are where a re-fetch from NECIS can help
(on 2021-08-27 KMA's source archive behind MARA lacks Z at 7 stations while N/E are present).

Writes (only with --apply): docs/station_inventory/tables/component_audit_<years>.csv, one row per partial station-day,
with flags for the two episodes (2021-07-01..2021-09-30, 2026-01-01..2026-02-28) and for the days of the 11 KMA events.

  python detection/tools/audit_components.py --years 2021,2026            (prints the summary)
  python detection/tools/audit_components.py --years 2021,2026 --apply    (also writes the CSV)
"""
import argparse, os, sys
from datetime import date, timedelta
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "registry"))
import gwangyang   # noqa: E402

LINKS = gwangyang.path(None, "dl_continuous")
STATIONS = gwangyang.path(None, "dl_station_table")
EPISODES = {"2021": ("2021-07-01", "2021-09-30"), "2026": ("2026-01-01", "2026-02-28")}


def kma_days():
    ev = pd.read_csv(os.path.join(ROOT, "registry", "events.csv"))
    col = next(c for c in ev.columns if "utc" in c.lower() or "time" in c.lower())
    return set(pd.to_datetime(ev[col]).dt.date)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--years", required=True); ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    years = [int(y) for y in a.years.split(",")]
    st = pd.read_csv(STATIONS, dtype=str).fillna("")
    kdays = kma_days()
    rows, tally = [], {}
    for r in st.itertuples():
        t0 = date.fromisoformat(r.Start); t1 = date.fromisoformat(r.End)
        if r.LinkUntil:
            t1 = min(t1, date.fromisoformat(r.LinkUntil))
        sdir = os.path.join(LINKS, r.Network, r.Code)
        have = {}
        for comp in "ZNE":
            cdir = os.path.join(sdir, f"{r.Sensor}{comp}.D")
            have[comp] = set(os.listdir(cdir)) if os.path.isdir(cdir) else set()
        for y in years:
            d = max(t0, date(y, 1, 1)); end = min(t1, date(y, 12, 31))
            while d <= end:
                doy = d.timetuple().tm_yday
                present = "".join(c for c in "ZNE" if f"{r.Network}.{r.Code}..{r.Sensor}{c}.D.{y}.{doy:03d}" in have[c])
                k = (y, "all" if len(present) == 3 else ("no data" if not present else "partial"))
                tally[k] = tally.get(k, 0) + 1
                if 0 < len(present) < 3:
                    ep = next((n for n, (s, e) in EPISODES.items() if s <= d.isoformat() <= e), "")
                    rows.append(dict(network=r.Network, station=r.Code, sensor=r.Sensor, date=d.isoformat(), year=y, doy=doy,
                                     present=present, missing="".join(c for c in "ZNE" if c not in present),
                                     episode=ep, kma_event_day=d in kdays))
                d += timedelta(days=1)
    D = pd.DataFrame(rows)
    print(f"link tree {LINKS}; station table {len(st)} rows; years {years}")
    for y in years:
        print(f"  {y}: complete {tally.get((y, 'all'), 0)}, partial {tally.get((y, 'partial'), 0)}, no data {tally.get((y, 'no data'), 0)} station-days")
    if len(D):
        print("partial station-days by missing set:", D.groupby(["year", "missing"]).size().to_dict())
        ep = D[D.episode != ""]
        print(f"partial station-days inside the episodes: {len(ep)} ({ep.groupby('episode').size().to_dict()}); on KMA-event days: {int(D.kma_event_day.sum())}")
        if len(ep):
            print(ep.sort_values(["date", "station"]).to_string(index=False))
    if a.apply:
        out = os.path.join(ROOT, "docs/station_inventory/tables", f"component_audit_{'_'.join(map(str, years))}.csv")
        D.to_csv(out, index=False); print("wrote", out)
    else:
        print("dry run: nothing written (--apply writes the CSV)")


if __name__ == "__main__":
    main()
