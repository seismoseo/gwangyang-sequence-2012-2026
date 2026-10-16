#!/usr/bin/env python3
"""Renewal step R2: the station table and the stream table of the renewed Haenam analysis.

From the station inventory (docs/decisions/station_inventory_20261011/tables/stations.csv: the user's station-set and
sensor decisions and sensor rule of 2026-10-11/12), one row per station code:

  detection/station_table/stations.csv   registry product dl_station_table; the columns the detection pipeline reads:
                                          Network, Code, Latitude, Longitude, Elevation, Sensor, Start, End
  registry/streams.csv                    the chosen sensor of each station for the whole window, its data period,
                                          role (co-located instruments and merged temporary codes keep their own codes),
                                          and its template-matching treatment

Sensor = the one chosen sensor for 2020-01-01 .. 2026-09-30. Start/End = its first and last data day in the window.
The continuous link tree (tools/link_continuous.py) links only this sensor, so every pipeline reading the tree uses it.
Elevation: KS and KG 0 m, as in the table used before the renewal (KMA's 2026 station list gives 0 m); JU from the
JNU StationXML, as before. The previous table is kept as stations_pre_renewal.csv.

  python detection/station_table/build_stations_renewal.py            # dry run: the table and its differences
  python detection/station_table/build_stations_renewal.py --apply
  python detection/station_table/build_stations_renewal.py --apply --streams-only   # registry/streams.csv only

After R3x (2026-10-13) the detection table stations.csv = this builder's table (kept as stations_r3.csv) plus the 2018-2019
predecessor codes with a LinkUntil column, written by detection/run_r3x_detection.sh prepare. A full --apply would drop them,
so it refuses when stations.csv has a LinkUntil column; refresh the stream table with --streams-only instead (2026-10-14:
streams.csv's BGDB note had kept the wiring-fault dates of before decision #33).
"""
import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # the project registry, found upward from this file
import argparse
import os
import shutil

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
INV = haenam.ROOT / "docs/station_inventory/tables/stations.csv"                       # GWANGYANG PORT: the Gwangyang inventory
OUT = os.path.join(HERE, "stations.csv")
OLD = os.path.join(HERE, "stations_pre_renewal.csv")
STREAMS = haenam.ROOT / "registry/streams.csv"
COLS = ["Network", "Code", "Latitude", "Longitude", "Elevation", "Sensor", "Start", "End"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--streams-only", action="store_true", help="write registry/streams.csv only (the detection table is left as it is)")
    a = ap.parse_args()
    inv = pd.read_csv(INV).sort_values("dist_km").reset_index(drop=True)
    old = pd.read_csv(OLD if os.path.exists(OLD) else OUT) if (os.path.exists(OLD) or os.path.exists(OUT)) else pd.DataFrame(columns=COLS)   # GWANGYANG PORT: no earlier table
    ju_elev = dict(zip(old[old.Network == "JU"].Code, old[old.Network == "JU"].Elevation))
    new = pd.DataFrame(dict(Network=inv.net, Code=inv.sta, Latitude=inv.lat.round(5), Longitude=inv.lon.round(5),
                            Elevation=[ju_elev.get(s, 0.0) if n == "JU" else 0.0 for n, s in zip(inv.net, inv.sta)],
                            Sensor=inv.chosen_sensor, Start=inv.first_day, End=inv.last_day))
    streams = pd.DataFrame(dict(net=inv.net, sta=inv.sta, site=inv.site, role=inv.role, dist_km=inv.dist_km, sensor=inv.chosen_sensor,
                                choice=inv.choice_status, first_day=inv.first_day, last_day=inv.last_day,
                                in_2021_seq=inv.in_2021_seq, in_2026_seq=inv.in_2026_seq, coords=inv.coords,   # GWANGYANG PORT: 2021 and 2026 episodes
                                template_matching=inv.s_component_rule, response_flags=inv.response_flags.fillna("")))
    # differences from the table used so far
    o = old.set_index(["Network", "Code"]); n = new.set_index(["Network", "Code"])
    added = sorted(set(n.index) - set(o.index)); removed = sorted(set(o.index) - set(n.index))
    both = sorted(set(n.index) & set(o.index))
    moved = [(k, round(((n.loc[k].Latitude - o.loc[k].Latitude) ** 2 + ((n.loc[k].Longitude - o.loc[k].Longitude) * 0.82) ** 2) ** 0.5 * 111.19, 2))
             for k in both]
    moved = [m for m in moved if m[1] > 0.05]
    sensor = [(k, o.loc[k].Sensor, n.loc[k].Sensor) for k in both if o.loc[k].Sensor != n.loc[k].Sensor]
    print(f"stations: {len(new)} ({new.Network.value_counts().to_dict()}); before: {len(old)}")
    print(f"added ({len(added)}): {', '.join('.'.join(k) for k in added)}")
    print(f"removed ({len(removed)}): {', '.join('.'.join(k) for k in removed) or 'none'}")
    print(f"moved > 50 m ({len(moved)}): {', '.join(f'{k[1]} {d} km' for k, d in moved) or 'none'}")
    print(f"sensor changed ({len(sensor)}): {', '.join(f'{k[1]} {a}->{b}' for k, a, b in sensor) or 'none'}")
    print(f"sensors: {new.Sensor.value_counts().to_dict()}")
    if not a.apply:
        print("DRY RUN: nothing written (add --apply)"); return 0
    if a.streams_only:
        streams.to_csv(STREAMS, index=False); print(f"wrote {STREAMS} (detection table left as it is)"); return 0
    if os.path.exists(OUT) and "LinkUntil" in pd.read_csv(OUT).columns:                 # GWANGYANG PORT: first build has no table
        raise SystemExit(f"{OUT} carries the R3x predecessor codes (LinkUntil): a full --apply would drop them; use --streams-only")
    if not os.path.exists(OLD) and os.path.exists(OUT):
        shutil.copy2(OUT, OLD)
    new[COLS].to_csv(OUT, index=False)
    streams.to_csv(STREAMS, index=False)
    print(f"wrote {OUT} and {STREAMS}; the previous table is {OLD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
