"""Write the well-resolved subset of the DL relocation as its own catalog (decimal-second origin times included).

Gate (all disclosed, all switchable):
  --cls resolved            notebook 06 class: bootstrap 95 % half-widths <= 100 m, >= 20 cc links, JN01 S-P consistent or unchecked
  --max-off 100             plane-normal offset from the canonical plane 96.27/67.53 below this (m)  [drops the far events]
  --clip 3                  iterative clip beyond clip x rms plane-normal distance of a robust SVD refit (0 = off)
  --max-err 0               additionally require horizontal and vertical 95 % half-widths below this (m); 0 = off
  --require-sp-check        keep only events whose JN01 S-P could be checked (P and S picks at JN01)
Default gate -> 366 events (the set the notebook 05/06 plane fits use).
Usage: python analysis/filter_well_resolved.py [--max-off 100 --clip 3 --max-err 20 --require-sp-check] [-o path]
"""
import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # the project registry, found upward from this file
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME = str(haenam.ROOT / 'catalogs/supplementary/Haenam_relocated_2026only_canonical_frame.csv')
FULL = str(haenam.ROOT / 'catalogs/supplementary/Haenam_relocated_2026only_481ev.csv')
OUT = str(haenam.ROOT / 'DL_detection/catalogs/haenam_dl_relocated_well_resolved.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", default="resolved"); ap.add_argument("--max-off", type=float, default=100.0)
    ap.add_argument("--clip", type=float, default=3.0); ap.add_argument("--max-err", type=float, default=0.0)
    ap.add_argument("--require-sp-check", action="store_true"); ap.add_argument("-o", "--out", default=OUT)
    a = ap.parse_args()
    fr = pd.read_csv(FRAME, comment="#", dtype={"eid": str, "template_match": str})
    full = pd.read_csv(FULL, dtype={"event_id": str, "template_match": str})
    d = fr[fr.cls == a.cls].copy(); n0 = len(d)
    d = d[d.off.abs() < a.max_off]; n1 = len(d)
    if a.clip > 0:
        X = np.column_stack([d.x, d.y, d.z]); keep = np.ones(len(X), bool)
        for _ in range(10):
            c0 = X[keep].mean(0); n = np.linalg.svd(X[keep] - c0, full_matrices=False)[2][-1]; dn = (X - c0) @ n
            rms = np.sqrt(np.mean(dn[keep] ** 2)); new = np.abs(dn) <= a.clip * rms
            if (new == keep).all():
                break
            keep = new
        d = d[keep]
    n2 = len(d)
    if a.max_err > 0:
        d = d[(np.hypot(d.ex95_m, d.ey95_m) < a.max_err) & (d.ez95_m < a.max_err)]
    if a.require_sp_check:
        d = d[d.sp_consistent.notna()]
    out = full[full.event_id.isin(d.eid)].merge(d[["eid", "al", "ud", "off", "x", "y", "z"]].rename(columns={"eid": "event_id", "x": "E_m", "y": "N_m", "z": "D_m"}), on="event_id")
    out = out.sort_values("origin_time_reloc")
    hdr = (f"# DL relocation, well-resolved subset: cls={a.cls}, |plane-normal offset| < {a.max_off:g} m, robust-plane clip {a.clip:g} x rms, "
           f"max 95 % half-width {a.max_err:g} m (0 = off), require S-P check {a.require_sp_check}; {n0} -> {n1} -> {n2} -> {len(out)} events; "
           f"al/ud/off/E_m/N_m/D_m in the canonical frame (plane 96.27/67.53, origin = Event 03)\n")
    with open(a.out, "w") as f:
        f.write(hdr); out.to_csv(f, index=False)
    print(hdr.strip()); print("->", a.out, f"({len(out)} rows, columns: {', '.join(out.columns[:6])}, ..., origin_time_reloc, al, ud, off)")


if __name__ == "__main__":
    main()
