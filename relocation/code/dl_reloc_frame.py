"""DL-relocated 2026 catalog in the CANONICAL fault frame (drop-in replacement for the 195-event `load_reloc()` used by
the 27.Haenam_sequence notebook suites).

Frame = relocation/frame_195ev_KSKG/plane195.npz (SVD plane strike 96.27, dip 67.53, origin = relocated Event 03) expressed in the
canonical run's local Cartesian (x east, y north, z down, metres). The DL positions are converted with the same
projection (offsets fitted on the full-precision reloc snapshot, rms 0.1-0.4 m) and shifted by the mean offset over
the events common to both catalogs (relative relocation constrains shape, not absolute position).

Constraint classes (notebook 06): resolved = bootstrap 95 % half-widths <= 100 m, >= 20 cc links, JN01 S-P consistent
(or unchecked); unresolved = enough links but large spread or S-P contradiction; weak = < 20 cc links.

Usage:  python analysis/dl_reloc_frame.py          # writes catalogs/haenam_dl_relocated_canonical_frame.csv
Library: load_reloc_dl(resolved_only=True) -> DataFrame with the load_reloc() columns
         (eid, t, mag, x, y, z, ex95_m, ey95_m, ez95_m, under_constrained) + al, ud, off, cls, cc_links, sp_consistent,
         template_match, ML_template.  `mag` = template ML where a canonical twin exists (115 events), else NaN.
"""
import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # the project registry, found upward from this file
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = str(haenam.ROOT / 'DL_detection/runs/hypodd_dl')
FINAL = str(haenam.ROOT / "relocation/provenance/hypodd_dl_2026only_481ev/final_locations.csv")
RELOC = str(haenam.ROOT / "relocation/provenance/hypodd_dl_2026only_481ev/hypoDD.reloc")
CATALOG = str(haenam.ROOT / 'catalogs/supplementary/Haenam_relocated_2026only_481ev.csv')
OUT = str(haenam.ROOT / 'catalogs/supplementary/Haenam_relocated_2026only_canonical_frame.csv')
import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # registry paths
PLANE = haenam.path(None, "plane195")
SNAP = haenam.path(None, "frame_195ev_KSKG") + "/hypoDD.reloc"
KCR = "/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation"
MIN_CC_LINKS = 20


def plane_frame(strike, dip):
    phi, dlt = np.radians(strike), np.radians(dip)
    u_s = np.array([np.sin(phi), np.cos(phi), 0.0])
    u_dd = np.array([np.cos(dlt) * np.sin(phi + np.pi / 2), np.cos(dlt) * np.cos(phi + np.pi / 2), np.sin(dlt)])

    def project(v):
        v = np.atleast_2d(v)
        return v @ u_s, -(v @ u_dd)
    return u_s, u_dd, project


def build():
    sys.path.insert(0, KCR)
    from pipeline.core import sumio
    fin = pd.read_csv(FINAL, dtype={"event_id": str}); fin["t"] = pd.to_datetime(fin.origin_time)
    cat = pd.read_csv(CATALOG, dtype={"event_id": str, "template_match": str})
    rl = sumio.read_reloc(RELOC); mf = pd.read_csv(str(haenam.ROOT / 'relocation/provenance/hypodd_dl_2026only_481ev/event_manifest.csv'), dtype={"event_id": str})
    cus = {200000 + int(r.event_idx): r.event_id for r in mf.itertuples()}; rl["event_id"] = rl.id.astype(int).map(cus)
    d = fin.merge(rl[["event_id", "nccp", "nccs"]], on="event_id", how="left")
    d = d.merge(cat[["event_id", "ML_template", "template_match", "sp_consistent", "sp_resid_ms"]], on="event_id", how="left")
    d["cc_links"] = d.nccp.fillna(0) + d.nccs.fillna(0); d["horiz95"] = np.hypot(d.ex95_m, d.ey95_m)
    bad = d.sp_consistent == 0
    d["cls"] = np.where(d.cc_links < MIN_CC_LINKS, "weak", np.where(d.under_constrained.astype(bool) | bad, "unresolved", "resolved"))
    # canonical local Cartesian: offsets from the full-precision snapshot
    rs = sumio.read_reloc(SNAP); lat0 = float(rs.lat.mean()); KX, KY = 111.19e3 * np.cos(np.radians(lat0)), 111.19e3
    cE = float((rs.x - rs.lon * KX).mean()); cN = float((rs.y - rs.lat * KY).mean()); cD = float((rs.z - rs.depth * 1e3).mean())
    res = np.column_stack([rs.lon * KX + cE - rs.x, rs.lat * KY + cN - rs.y, rs.depth * 1e3 + cD - rs.z])
    rms = np.sqrt((res ** 2).mean(0)); assert rms.max() < 1.0, f"projection residual {rms} m"
    V = np.column_stack([d.longitude * KX + cE, d.latitude * KY + cN, d.depth_km * 1e3 + cD])
    pz = np.load(PLANE, allow_pickle=True)
    can = pd.DataFrame(dict(event_id=[str(e) for e in pz["eid"]], E=pz["E"].astype(float), N=pz["N"].astype(float), D=pz["D"].astype(float)))
    com = d[(d.cls == "resolved") & d.template_match.notna()].merge(can, left_on="template_match", right_on="event_id", suffixes=("", "_c"))
    Vc = np.column_stack([com.longitude * KX + cE, com.latitude * KY + cN, com.depth_km * 1e3 + cD])
    shift = (Vc - np.column_stack([com.E, com.N, com.D])).mean(0)
    V = V - shift
    strike, dip, normal, origin = float(pz["strike"]), float(pz["dip"]), np.asarray(pz["normal"], float), np.asarray(pz["origin"], float)
    u_s, u_dd, project = plane_frame(strike, dip)
    al, ud = project(V - origin); off = (V - origin) @ normal
    out = pd.DataFrame(dict(eid=d.event_id, t=d.t, mag=d.ML_template, x=V[:, 0], y=V[:, 1], z=V[:, 2], al=al, ud=ud, off=off,
                            ex95_m=d.ex95_m, ey95_m=d.ey95_m, ez95_m=d.ez95_m, under_constrained=d.under_constrained.astype(bool),
                            cls=d.cls, cc_links=d.cc_links.astype(int), sp_consistent=d.sp_consistent, template_match=d.template_match,
                            ML_template=d.ML_template, lat=d.latitude, lon=d.longitude, depth_km=d.depth_km))
    meta = dict(strike=strike, dip=dip, shift_E=shift[0], shift_N=shift[1], shift_D=shift[2], n_common=len(com), proj_rms_m=rms.max())
    return out, meta


def load_reloc_dl(resolved_only=True, path=OUT):
    "DL 2026 catalog in the canonical frame with the load_reloc() columns (+ cls, al, ud, off)."
    df = pd.read_csv(path, comment="#", dtype={"eid": str, "template_match": str}, parse_dates=["t"])
    return df[df.cls == "resolved"].reset_index(drop=True) if resolved_only else df


if __name__ == "__main__":
    out, meta = build()
    hdr = ("# DL 2026 relocation in the canonical frame (plane195: strike %.2f dip %.2f, origin = Event 03); DL shifted by "
           "E %+.0f N %+.0f D %+.0f m over %d common resolved events; projection rms %.2f m\n" % (
               meta["strike"], meta["dip"], meta["shift_E"], meta["shift_N"], meta["shift_D"], meta["n_common"], meta["proj_rms_m"]))
    with open(OUT, "w") as f:
        f.write(hdr); out.to_csv(f, index=False)
    print(hdr.strip()); print(out.cls.value_counts().to_dict(), "->", OUT)
    r = out[out.cls == "resolved"]
    print(f"resolved: {len(r)}; with ML: {int(r.mag.notna().sum())}; along-strike 5-95 % {r.al.quantile(.05):+.0f}..{r.al.quantile(.95):+.0f} m; off-plane median |off| {r.off.abs().median():.0f} m")
