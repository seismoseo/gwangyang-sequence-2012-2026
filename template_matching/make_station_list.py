"""GWANGYANG (decisions G-D4, G-D15; 60 km: see G-D4): the template-matching scan stations, from the frozen station table.

Every code of detection/station_table/stations.csv within RADIUS_KM of the cluster centroid, one chosen sensor each
(the table's Sensor), P on Z and S on N/E at every station (horizontals=1; templates are turned into each scanned day's
orientation by tm_orientation.py). Temporary codes (registry/streams.csv rows whose sta differs from their site, e.g. GWYBT)
are left out: the scan reads them as their site (config.TEMP_CODES). Predecessor codes (YSU, NAH, ...) stay separate
stations: a template carries picks only at the codes that recorded it.
Prints the number of stations per year for RADIUS_KM and for the plan's ~60 km (G-D4 asks for both).

  python template_matching/make_station_list.py            (dry run)
  python template_matching/make_station_list.py --apply    (writes template_matching/stations_G7_60km.csv)
"""
import argparse, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "registry"))
import gwangyang   # noqa: E402

CENTER = (35.0000, 127.7286)          # cluster centroid (lat, lon)
RADIUS_KM = 60.0
COMPARE_KM = 50.0
OUT = os.path.join(HERE, "stations_G7_60km.csv")


def dist_km(lat, lon):
    la0, lo0 = np.radians(CENTER); la, lo = np.radians(lat), np.radians(lon)
    a = np.sin((la - la0) / 2) ** 2 + np.cos(la0) * np.cos(la) * np.sin((lo - lo0) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true"); a = ap.parse_args()
    st = pd.read_csv(gwangyang.path(None, "dl_station_table"), dtype={"LinkUntil": str})
    streams = pd.read_csv(os.path.join(ROOT, "registry", "streams.csv"), dtype=str)
    temp = set(zip(streams.net[streams.sta != streams.site], streams.sta[streams.sta != streams.site]))
    st["dist_km"] = dist_km(st.Latitude.to_numpy(), st.Longitude.to_numpy())
    st["temporary"] = [(n, c) in temp for n, c in zip(st.Network, st.Code)]
    st["last"] = [min(e, u) if isinstance(u, str) and u else e for e, u in zip(st.End, st.LinkUntil)]
    years = range(2012, 2027)
    def per_year(r):
        s = st[(st.dist_km <= r) & ~st.temporary]
        return [int(((s.Start <= f"{y}-12-31") & (s["last"] >= f"{y}-01-01")).sum()) for y in years]
    print("stations per year (codes with data in the year; temporary codes excluded):")
    print(pd.DataFrame({f"<= {RADIUS_KM:.0f} km": per_year(RADIUS_KM), f"<= {COMPARE_KM:.0f} km": per_year(COMPARE_KM)}, index=list(years)).T.to_string())
    sel = st[(st.dist_km <= RADIUS_KM) & ~st.temporary].sort_values("dist_km")
    out = pd.DataFrame(dict(network=sel.Network, station=sel.Code, horizontals=1, s_exclude=0,
                            note=[f"{d:.1f} km; {s}; {b}..{e}" for d, s, b, e in zip(sel.dist_km, sel.Sensor, sel.Start, sel["last"])]))
    print(f"\n{len(out)} scan stations within {RADIUS_KM:.0f} km:"); print(out.to_string(index=False))
    left = st[(st.dist_km <= RADIUS_KM) & st.temporary]
    if len(left):
        print("temporary codes read as their site:", ", ".join(f"{n}.{c}" for n, c in zip(left.Network, left.Code)))
    if a.apply:
        with open(OUT, "w") as fh:
            fh.write(f"# Gwangyang G7 scan stations (decision G-D4): the {len(out)} codes of detection/station_table/stations.csv within {RADIUS_KM:.0f} km of\n"
                     "# the cluster centroid, temporary codes excluded (read as their site). P on Z and S on N/E at every station (templates turned).\n")
            out.to_csv(fh, index=False)
        print("wrote", OUT)
    else:
        print("dry run: nothing written (--apply writes", OUT, ")")


if __name__ == "__main__":
    main()
