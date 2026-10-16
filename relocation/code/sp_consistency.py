"""JN01 S-P consistency check of relocated hypocenters (independent of HypoDD).

The S-P time at a station changes by K_SP = 1/Vs - 1/Vp seconds per km of hypocentral distance. For every event
with P and S picks at the reference station we compare the OBSERVED change of S-P relative to the reference event
(the KMA M2.1) with the change PREDICTED by the relocated positions. A residual beyond TOL_MS means the relocated
position is inconsistent with the picks at the best near station: with the kim1983 velocities at 20 km depth
(Vp 6.38 km/s, Vp/Vs 1.73) 100 ms corresponds to about 0.9 km.

Usage:  python analysis/sp_consistency.py            # adds sp_* columns to catalogs/haenam_dl_relocated.csv
Library: sp_residuals(df, wf_dir, station_csv, ref_event) -> df + columns
    sp_obs_s (observed S-P), dsp_obs_ms, dsp_pred_ms, sp_resid_ms (obs - pred), sp_consistent (True/False, NaN = no picks)
"""
import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # the project registry, found upward from this file
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = str(haenam.ROOT / "relocation/provenance/hypodd_dl_2026only_481ev/final_locations.csv")
CATALOG = str(haenam.ROOT / 'catalogs/supplementary/Haenam_relocated_2026only_481ev.csv')
WF = haenam.path(None, 'wf_dl_2026only_481ev')
STATIONS = haenam.path(None, 'dl_station_table')
REF_EVENT = "20260814084339"          # KMA M2.1
STATION = "JN01"
VP_SRC, VPVS = 6.38, 1.73             # kim1983 layer below 15 km (the cluster is at ~20 km)
K_SP = 1.0 / (VP_SRC / VPVS) - 1.0 / VP_SRC      # s per km of hypocentral distance (0.114)
TOL_MS = 100.0                        # ~0.9 km


def observed_sp(event_ids, wf_dir=WF, station=STATION):
    """S-P (s) from the exported SAC headers (a = P pick, t0 = S pick) on the vertical trace; missing -> NaN."""
    from obspy import read
    out = {}
    for eid in event_ids:
        fl = sorted(glob.glob(os.path.join(wf_dir, str(eid), f"*{station}*Z*.sac")))
        if not fl:
            continue
        h = read(fl[0], headonly=True)[0].stats.sac
        a, t0 = h.get("a", -12345.0), h.get("t0", -12345.0)
        if a > -12345 and t0 > -12345:
            out[str(eid)] = float(t0 - a)
    return out


def sp_residuals(df, wf_dir=WF, station_csv=STATIONS, ref_event=REF_EVENT, station=STATION, tol_ms=TOL_MS):
    st = pd.read_csv(station_csv)
    s = st[st.Code == station].iloc[0]
    cl = np.cos(np.radians(float(s.Latitude)))

    def dist_km(lat, lon, dep_km):
        return np.sqrt(((np.asarray(lon) - s.Longitude) * 111.19 * cl) ** 2 + ((np.asarray(lat) - s.Latitude) * 111.19) ** 2
                       + (np.asarray(dep_km) + float(s.Elevation) / 1000.0) ** 2)

    d = df.copy()
    d["event_id"] = d.event_id.astype(str)
    sp = observed_sp(d.event_id, wf_dir, station)
    if ref_event not in sp:
        raise RuntimeError(f"reference event {ref_event} has no {station} P/S picks")
    ref = d[d.event_id == ref_event].iloc[0]
    d["sp_obs_s"] = d.event_id.map(sp)
    d["dsp_obs_ms"] = (d.sp_obs_s - sp[ref_event]) * 1000.0
    d["dsp_pred_ms"] = (dist_km(d.latitude, d.longitude, d.depth_km) - dist_km(ref.latitude, ref.longitude, ref.depth_km)) * K_SP * 1000.0
    d["sp_resid_ms"] = d.dsp_obs_ms - d.dsp_pred_ms
    d["sp_consistent"] = np.where(d.sp_obs_s.isna(), np.nan, (d.sp_resid_ms.abs() <= tol_ms).astype(float))
    return d


def main():
    fin = pd.read_csv(FINAL, dtype={"event_id": str})
    d = sp_residuals(fin)
    ok = d.sp_obs_s.notna()
    print(f"[sp] {station_line()} picks on {int(ok.sum())}/{len(d)} events; |residual| median {d.loc[ok, 'sp_resid_ms'].abs().median():.0f} ms, "
          f"90th pct {d.loc[ok, 'sp_resid_ms'].abs().quantile(.9):.0f} ms; inconsistent (> {TOL_MS:.0f} ms): {int((d.sp_consistent == 0).sum())}")
    cols = ["event_id", "sp_obs_s", "dsp_obs_ms", "dsp_pred_ms", "sp_resid_ms", "sp_consistent"]
    if os.path.exists(CATALOG):
        cat = pd.read_csv(CATALOG, dtype={"event_id": str})
        cat = cat.drop(columns=[c for c in cols[1:] if c in cat.columns]).merge(d[cols].round(1), on="event_id", how="left")
        cat.to_csv(CATALOG, index=False)
        print(f"[sp] columns {cols[1:]} written to {CATALOG} ({len(cat)} rows)")
    else:
        print("[sp] no project catalog yet:", CATALOG)


def station_line():
    return f"{STATION} S-P (K = {K_SP:.3f} s/km, tolerance {TOL_MS:.0f} ms ~ {TOL_MS / 1000 / K_SP:.1f} km)"


if __name__ == "__main__":
    main()
