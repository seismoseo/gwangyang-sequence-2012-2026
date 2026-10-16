#!/usr/bin/env python3
"""GWANGYANG PORT of relocation/code/adopt_r4.py (Haenam renewal R4 adoption): the 25 m reference set and the fault plane
of a relocation run, with the same definitions, but anchored on the 2021 Gwangyang ML 2.2 and with Gwangyang eras.
No counts or event ids are asserted: they are reported.

  reference set  every relocated event whose finite bootstrap 95% half-widths are <= 25 m in east, north and vertical
                 (Haenam full_period_publication: pass25)
  focused set    the reference events within 1 km of the 2021 ML 2.2 (KMA 2021-08-28 07:03:22 KST = 2021-08-27 22:03:22 UTC),
                 the "patch view"; every reference event outside it is listed
  plane          joint_science.plane() (pasted verbatim from the Haenam tree): unweighted SVD of (east, north, depth) of the
                 focused set; 95% intervals from 1000 event resamples, seed 20260922
Eras: before the 2021 episode, the 2021 episode (2021-07-01 .. 2021-10-01), the interval, the 2026 episode
(2026-01-01 .. 2026-03-01), after it. The episode bounds are the KMA months of the 11 listed events, widened to whole months.

Differences from adopt_r4.py: anchor event and eras (above); the default output is the registry product reference_25m_dir;
--apply is a real option (in adopt_r4.py of 2026-10-15 it sits inside the comment of the --out line, so a.apply fails).

  python relocation/code/adopt_gwangyang.py --run-root relocation/hypodd_dl_2012-2026            (dry run)
  python relocation/code/adopt_gwangyang.py --run-root relocation/hypodd_dl_2012-2026 --apply
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "registry"))
import gwangyang   # noqa: E402

MAIN_TIME = pd.Timestamp("2021-08-27 22:03:22")
ERAS = [("Before 2021", "2012-12-01", "2021-07-01"), ("2021 episode", "2021-07-01", "2021-10-01"),
        ("Interval", "2021-10-01", "2026-01-01"), ("2026 episode", "2026-01-01", "2026-03-01"),
        ("After 2026 episode", "2026-03-01", "2026-11-01")]


def plane(X):
    center=np.mean(X,axis=0);_,sv,v=np.linalg.svd(X-center,full_matrices=False);n=v[-1]
    if n[2]>0:n=-n
    dip=np.degrees(np.arccos(np.clip(-n[2],-1,1)));strike=(np.degrees(np.arctan2(n[0],n[1]))-90)%360
    s=np.array([np.sin(np.radians(strike)),np.cos(np.radians(strike)),0]);h=np.array([np.cos(np.radians(strike)),-np.sin(np.radians(strike)),0]);down=h*np.cos(np.radians(dip))+np.array([0,0,np.sin(np.radians(dip))])
    return dict(center=center,normal=n,strike=strike,dip=dip,s=s,h=h,down=down,rms=float(np.sqrt(np.mean(((X-center)@n)**2))),sv=sv)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-root", required=True)
    ap.add_argument("--out", default=str(gwangyang.path(None, "reference_25m_dir")))
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    run = os.path.join(ROOT, a.run_root) if not os.path.isabs(a.run_root) else a.run_root
    br = os.path.join(run, "2.HypoDD/02.dt.cc")
    d = pd.read_csv(os.path.join(run, "haenam_dl_relocated.csv"), dtype={"event_id": str, "template_match": str})
    m = pd.read_csv(os.path.join(run, "event_manifest.csv"), dtype={"event_id": str}); m["id"] = 200000 + m.event_idx
    e = pd.read_csv(os.path.join(br, "bootstrap_errors.csv"), comment="#").merge(m[["id", "event_id"]], on="id", validate="one_to_one")
    d = d.merge(e[["event_id", "id", "ex95", "ey95", "ez95", "n_boot"]], on="event_id", validate="one_to_one", suffixes=("", "_raw"))
    d["time"] = pd.to_datetime(d.origin_time)
    d["era"] = np.select([(d.time >= pd.Timestamp(s)) & (d.time < pd.Timestamp(t)) for _, s, t in ERAS], [n for n, _, _ in ERAS], default="?")
    errs = d[["ex95", "ey95", "ez95"]].to_numpy()
    d["pass25"] = np.isfinite(errs).all(axis=1) & (errs <= 25).all(axis=1)
    d["max95_m"] = np.max(errs, axis=1)
    # local frame at the 2021 ML 2.2; found by time, as its DL id may move by a second
    i0 = (d.time - MAIN_TIME).abs().idxmin(); assert abs((d.time[i0] - MAIN_TIME).total_seconds()) <= 3, "2021 ML 2.2 not relocated"
    lat0, lon0, z0 = float(d.latitude[i0]), float(d.longitude[i0]), float(d.depth_km[i0])
    km_lat = 111.19; km_lon = km_lat * np.cos(np.radians(lat0))
    d["east"] = (d.longitude - lon0) * km_lon * 1000; d["north"] = (d.latitude - lat0) * km_lat * 1000; d["depth_m"] = d.depth_km * 1000
    d["distance_m"] = np.linalg.norm(np.column_stack([d.east, d.north, (d.depth_km - z0) * 1000]), axis=1)
    ref = d[d.pass25].copy()
    foc = ref[ref.distance_m <= 1000].copy()
    off = ref[ref.distance_m > 1000]
    X = foc[["east", "north", "depth_m"]].to_numpy(); p = plane(X); rng = np.random.default_rng(20260922)
    boot = np.array([[b["strike"], b["dip"]] for b in [plane(X[rng.integers(0, len(X), len(X))]) for _ in range(1000)]])
    p["strike_ci"] = np.percentile(boot[:, 0], [2.5, 97.5]); p["dip_ci"] = np.percentile(boot[:, 1], [2.5, 97.5])
    by_era = pd.DataFrame(dict(relocated=d.groupby("era").size(), reference_25m=ref.groupby("era").size(), focused=foc.groupby("era").size())).reindex([n for n, _, _ in ERAS]).fillna(0).astype(int)
    print(f"run {a.run_root}: submitted {len(m)}, relocated {len(d)}, 25 m reference {len(ref)}, focused (within 1 km of the 2021 ML 2.2 {d.event_id[i0]}) {len(foc)}")
    print(by_era.to_string())
    if (d.era == "?").any():
        print(f"events outside every era: {int((d.era == '?').sum())}")
    if len(off):
        print("reference events outside the patch (excluded from the plane):")
        print(off[["event_id", "origin_time", "latitude", "longitude", "depth_km", "distance_m", "max95_m"]].to_string(index=False))
    print(f"plane: strike {p['strike']:.2f} (95% {p['strike_ci'][0]:.2f}-{p['strike_ci'][1]:.2f}), dip {p['dip']:.2f} "
          f"(95% {p['dip_ci'][0]:.2f}-{p['dip_ci'][1]:.2f}), RMS {p['rms']:.1f} m, n {len(foc)}")
    if a.apply:
        os.makedirs(a.out, exist_ok=True)
        ref.to_csv(os.path.join(a.out, "reference_25m.csv"), index=False); foc.to_csv(os.path.join(a.out, "focused_reference.csv"), index=False)
        np.savez(os.path.join(a.out, "whole_sequence_plane.npz"), **p, fit_event_ids=foc.event_id.to_numpy(dtype=str), event_resample_orientations=boot)
        json.dump(dict(run_root=a.run_root, submitted=len(m), relocated=len(d), reference_25m=len(ref), focused=len(foc),
                       main_event=str(d.event_id[i0]), excluded_off_patch=off.event_id.tolist(), by_era=by_era.to_dict(),
                       strike=float(p["strike"]), dip=float(p["dip"]), strike_ci=list(map(float, p["strike_ci"])), dip_ci=list(map(float, p["dip_ci"])),
                       rms_m=float(p["rms"])), open(os.path.join(a.out, "summary.json"), "w"), indent=1)
        print("written to", a.out)
    else:
        print("dry run: nothing written (--apply writes", a.out, ")")


if __name__ == "__main__":
    main()
