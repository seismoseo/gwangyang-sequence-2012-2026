#!/usr/bin/env python3
"""R4 adoption: the 25 m reference set and the whole-sequence fault plane of a relocation run, with the project's own
definitions (analysis/seismicity/full_period_publication.py load(); joint_science.py plane() and load()), but no counts
or event ids asserted: they are reported.

  reference set  every relocated event whose finite bootstrap 95% half-widths are <= 25 m in east, north and vertical
                 (full_period_publication: pass25)
  focused set    the reference events within 1 km of the M21 event (20260814 08:43:39), the "patch view" of
                 full_period_publication. The old run excluded its one event outside it (2023-05-22) by id; here the rule
                 itself is applied and every excluded event is listed.
  plane          joint_science.plane(): unweighted SVD of (east, north, depth) of the focused set; 95% intervals from 1000
                 event resamples, seed 20260922
Eras: 2018-2019 (the R3x extension), 2020 period (to 2020-08-01), quiescence, 2026 sequence (from 2026-08-14).

  python relocation/code/adopt_r4.py --run-root relocation/hypodd_dl_renewal_R4 [--out relocation/provenance/adoption_r4/reference_25m_R4] [--apply]
  python relocation/code/adopt_r4.py --run-root relocation/hypodd_dl_2020-2026      (check: must give 607 / 606 and the
                                                                                      adopted plane 97.31 / 64.22)
Without --apply nothing is written.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "registry")); sys.path.insert(0, os.path.join(ROOT, "analysis/seismicity"))
import joint_science as J      # noqa: E402  (plane())

M21_TIME = pd.Timestamp("2026-08-14 08:43:39")
ERAS = [("2018-2019", "2018-01-01", "2020-01-01"), ("2020 period", "2020-01-01", "2020-08-01"),
        ("Quiescence", "2020-08-01", "2026-08-14"), ("2026 sequence", "2026-08-14", "2026-10-01")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-root", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "relocation/provenance/adoption_r4/reference_25m_R4"))   # moved from catalogs/ 2026-10-15 (the canonical reference is catalogs/reference_25m_646ev); ap.add_argument("--apply", action="store_true")
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
    # local frame at the M21 event (full_period_publication); found by time, as its DL id may move by a second
    i21 = (d.time - M21_TIME).abs().idxmin(); assert abs((d.time[i21] - M21_TIME).total_seconds()) <= 2, "M21 not relocated"
    lat0, lon0, z0 = float(d.latitude[i21]), float(d.longitude[i21]), float(d.depth_km[i21])
    km_lat = 111.19; km_lon = km_lat * np.cos(np.radians(lat0))
    d["east"] = (d.longitude - lon0) * km_lon * 1000; d["north"] = (d.latitude - lat0) * km_lat * 1000; d["depth_m"] = d.depth_km * 1000
    d["distance_m"] = np.linalg.norm(np.column_stack([d.east, d.north, (d.depth_km - z0) * 1000]), axis=1)
    ref = d[d.pass25].copy()
    foc = ref[ref.distance_m <= 1000].copy()
    off = ref[ref.distance_m > 1000]
    X = foc[["east", "north", "depth_m"]].to_numpy(); p = J.plane(X); rng = np.random.default_rng(20260922)
    boot = np.array([[b["strike"], b["dip"]] for b in [J.plane(X[rng.integers(0, len(X), len(X))]) for _ in range(1000)]])
    p["strike_ci"] = np.percentile(boot[:, 0], [2.5, 97.5]); p["dip_ci"] = np.percentile(boot[:, 1], [2.5, 97.5])
    by_era = pd.DataFrame(dict(relocated=d.groupby("era").size(), reference_25m=ref.groupby("era").size(), focused=foc.groupby("era").size())).reindex([n for n, _, _ in ERAS]).fillna(0).astype(int)
    print(f"run {a.run_root}: submitted {len(m)}, relocated {len(d)}, 25 m reference {len(ref)}, focused (within 1 km of M21 {d.event_id[i21]}) {len(foc)}")
    print(by_era.to_string())
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
                       m21_event=str(d.event_id[i21]), excluded_off_patch=off.event_id.tolist(), by_era=by_era.to_dict(),
                       strike=float(p["strike"]), dip=float(p["dip"]), strike_ci=list(map(float, p["strike_ci"])), dip_ci=list(map(float, p["dip_ci"])),
                       rms_m=float(p["rms"])), open(os.path.join(a.out, "summary.json"), "w"), indent=1)
        print("written to", a.out)


if __name__ == "__main__":
    main()
