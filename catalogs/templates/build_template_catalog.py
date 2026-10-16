"""GWANGYANG PORT of catalogs/templates/build_template_catalog_R4.py (Haenam renewal R5): the template catalog for template
matching = the focused 25 m reference set of the Gwangyang relocation (relocation/code/adopt_gwangyang.py).

Source: <reference_25m_dir>/focused_reference.csv (every relocated event whose finite bootstrap 95% half-widths are <= 25 m
in east, north and vertical, within 1 km of the 2021 ML 2.2). Picks: the run's own per-event SAC exports
(<run>/waveforms_100km/<event_id>/, SAC a = P, t0 = S). ML: the G6 local magnitude where the registry has an ml_table
product (column ml, source column ml_source), else empty; the scan does not use it, only the detection magnitudes after the
scan do. cuspid = 700000 + row index (chronological; Haenam R4 used 500000).
Output: catalogs/templates/Gwangyang_templates_ref25.csv (registry tm_templates_195, the product name kept from the Haenam
tree so template_matching/config.py needs no new key), in the template_matching/templates.py format.

  python catalogs/templates/build_template_catalog.py --run-root relocation/hypodd_dl_2012-2026            (dry run)
  python catalogs/templates/build_template_catalog.py --run-root relocation/hypodd_dl_2012-2026 --apply
"""
import argparse, os, sys, pathlib
import numpy as np
import pandas as pd
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "gwangyang.py").is_file()) / "registry")); import gwangyang   # noqa: E402

REF = pathlib.Path(gwangyang.path(None, "reference_25m_dir")) / "focused_reference.csv"
OUT = pathlib.Path(gwangyang.path(None, "tm_templates_195"))
CUSPID_OFFSET = 700000
SEQ = {"Before 2021": "pre2021", "2021 episode": "2021", "Interval": "interval", "2026 episode": "2026",
       "After 2026 episode": "post2026"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-root", required=True); ap.add_argument("--apply", action="store_true"); a = ap.parse_args()
    run = gwangyang.ROOT / a.run_root if not os.path.isabs(a.run_root) else pathlib.Path(a.run_root)
    WF = run / "waveforms_100km"
    if not REF.exists():
        raise SystemExit(f"{REF} does not exist yet: run relocation/code/adopt_gwangyang.py --run-root {a.run_root} --apply first")
    d = pd.read_csv(REF, dtype={"event_id": str})
    d["origin_time_UTC"] = pd.to_datetime(d.origin_time); d = d.sort_values("origin_time_UTC").reset_index(drop=True)
    d["cuspid"] = CUSPID_OFFSET + np.arange(len(d))
    d["horiz_err95_m"] = np.hypot(d.ex95_m, d.ey95_m); d["vert_err95_m"] = d.ez95_m
    unknown = sorted(set(d.era) - set(SEQ))
    assert not unknown, f"eras without a sequence name: {unknown}"
    d["sequence"] = d.era.map(SEQ); d["confidence"] = "high_conf"
    if "ml_table" in gwangyang.products():
        mls = pd.read_csv(gwangyang.path(None, "ml_table"), dtype={"event_id": str}).set_index("event_id")
        d["ML"] = d.event_id.map(mls.ml); d["ml_source"] = np.where(d.ML.notna(), "ml " + d.event_id.map(mls.ml_source).fillna(""), "none")
    else:
        d["ML"] = np.nan; d["ml_source"] = "none"
        print("no ml_table product in the registry yet: ML left empty (the scan does not use it)")
    out = d[["cuspid", "event_id", "origin_time_UTC", "latitude", "longitude", "depth_km", "ML", "ml_source",
             "horiz_err95_m", "vert_err95_m", "cc_links", "ct_links", "under_constrained", "sequence", "confidence"]]
    out = out.rename(columns={"cc_links": "n_cc_links", "ct_links": "n_ct_links"})
    missing = [e for e in out.event_id if not (WF / e).is_dir()]
    assert not missing, f"{len(missing)} events without SAC exports in {WF}: {missing[:5]}"
    print(f"{len(out)} templates from {REF}:"); print(out.groupby("sequence").size().to_string())
    print("median horiz95 / vert95 (m):", round(out.horiz_err95_m.median(), 1), "/", round(out.vert_err95_m.median(), 1))
    if a.apply:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUT.with_suffix(".tmp"); out.to_csv(tmp, index=False); os.replace(tmp, OUT); print("wrote", OUT)   # atomic: a reader never sees a partial file
    else:
        print("dry run: nothing written (--apply writes", OUT, ")")


if __name__ == "__main__":
    main()
