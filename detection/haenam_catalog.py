"""Build the Haenam DL catalog from the HYPOINVERSE .sum files (all years present), with QC TIERS,
distances, and cross-matches. Port of 21.2024_Buan_precursor/analysis/precursor_catalog.py.

For every located event:
  quality_gate    the study's one quality rule (user, 2026-10-13) = the relocation's gate config.HAENAM_QC:
                  gap<180, RMS<1.0, ERH/ERZ<5 km, N>=8
  relocation_set  R4's input (relocation/code/relocation_selection.py): quality_gate AND within 5 km of the cluster
                  center AND depth >= 10 km (first of a same-second pair)
  tier            legacy label A (gap<180, RMS<0.7, ERH/ERZ<5 km, N>=8) / B (gap<300, RMS<1.0, ERH/ERZ<10, N>=6) / '-';
                  nothing in the study reads it
  dist_haenam_km  epicentral distance to the Haenam cluster (config.HAENAM_EPICENTER)
  in_haenam_box   inside config.HAENAM_BOX (candidates for the relocation fold-back)
  nearest_sta_km  distance to the nearest station with data that day (sensitivity proxy)
  KMA_match       'M<mag>' if a KMA event lies within KMA_TOL_S (entire KMA catalog + the 2026 Haenam KMA
                  list used by the relocation; both KST -> UTC), else '-'
  template_match  event_id of the relocated (template) catalog within TPL_TOL_S, else '-'
  unreported      no KMA match
  promote         tier A/B AND in_haenam_box AND no template match  -> stage_for_augment candidates
Writes catalogs/haenam_dl_catalog.csv (one row per located event, all years, sorted by time).

Run:  conda run -n base python analysis/haenam_catalog.py            (after ./run_haenam.sh locate)
"""
import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # the project registry, found upward from this file
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, str(haenam.ROOT / 'detection/pipeline'))
import config  # noqa: E402
from sumio import read_sum  # noqa: E402
import pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # registry paths

MODEL = "phasenet_plus"
VELMODEL = "kim1983"
# Renewal (2026-10-13): the newest KMA lists, both in-project copies. The national catalog (KST) runs to 2026-03-17; the
# Haenam list (KMA, 207 events, already UTC) runs to 2026-09-16, so Haenam events after March 2026 are still matched.
KMA_CSV = haenam.path(None, "kma_national_catalog")                              # GWANGYANG PORT: KMA national catalog (KST) to 2026-03-09
KMA_2026_CSV = haenam.path(None, "kma_necis_domestic")                           # GWANGYANG PORT: NECIS domestic catalog (UTC) to 2026-10-13
KMA_2026_IS_UTC = True
TEMPLATES_CSV = haenam.path(None, "tm_templates_195")
OUT = haenam.path(None, 'dl_catalog')
KMA_TOL_S = 12.0
TPL_TOL_S = 6.0


def _dist_km(lat, lon, lat0, lon0):
    return 111.2 * np.hypot(lat - lat0, (lon - lon0) * np.cos(np.radians(lat0)))


def qc_tier(r):
    def ok(q):
        return (r.erh < q["erh"] and r.erz < q["erz"] and r.gap < q["gap"]
                and r.num > q["num"] and r.rms < q["rms"])
    if ok(config.HAENAM_QC_A):
        return "A"
    if ok(config.HAENAM_QC_B):
        return "B"
    return "-"


def kma_events():
    k = pd.read_csv(KMA_CSV, parse_dates=["time"], encoding="utf-8-sig")
    k["time"] = k["time"] - pd.Timedelta(hours=9)
    k["magnitude"] = pd.to_numeric(k["magnitude"], errors="coerce")
    k = k[["time", "latitude", "longitude", "magnitude"]]
    if os.path.exists(KMA_2026_CSV):
        k2 = pd.read_csv(KMA_2026_CSV, encoding="utf-8-sig")               # GWANGYANG PORT: NECIS columns (origin_utc)
        k2 = pd.DataFrame(dict(time=pd.to_datetime(k2.origin_utc), latitude=k2.latitude, longitude=k2.longitude,
                               magnitude=pd.to_numeric(k2.magnitude, errors="coerce")))
        k = pd.concat([k, k2], ignore_index=True).drop_duplicates("time")
    return k.sort_values("time").reset_index(drop=True)


def nearest(t, ref, tol_s):
    d = (ref.time - t).abs()
    if len(d) and d.min() <= pd.Timedelta(seconds=tol_s):
        return d.idxmin()
    return None


def main():
    # years and output (2026-10-13): the default is the 2020-2026 product (registry dl_catalog) whatever other years are
    # located; the 2018-2026 merge with the R3x extension is its own product (registry dl_catalog_2018_2026)
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--years", default="2012-2026"); ap.add_argument("--out", default=None)   # GWANGYANG PORT
    a = ap.parse_args()
    y0, y1 = (int(x) for x in a.years.split("-"))
    global OUT
    if a.out:
        OUT = a.out
    sums = sorted(f for f in glob.glob(f"{config.velmodel_dir(MODEL, VELMODEL)}/{config.REGION_NAME}*.sum")
                  if y0 <= int(os.path.basename(f)[len(config.REGION_NAME):-4]) <= y1)
    if not sums:
        sys.exit(f"no .sum files under {config.velmodel_dir(MODEL, VELMODEL)} — run ./run_haenam.sh locate first")
    df = pd.concat([read_sum(s) for s in sums], ignore_index=True)
    df = df[df.time.notna() & df.lat.notna()].reset_index(drop=True)

    df["tier"] = df.apply(qc_tier, axis=1)
    la0, lo0 = config.HAENAM_EPICENTER
    df["dist_haenam_km"] = _dist_km(df.lat, df.lon, la0, lo0).round(2)
    b = config.HAENAM_BOX
    df["in_haenam_box"] = (df.lat.between(b[0], b[1]) & df.lon.between(b[2], b[3]))
    # nearest station WITH DATA that day: a station counts when the link tree holds its Z day file (exact for both the
    # 2020-2026 run and the 2018-2019 extension, whose table also lists the predecessor codes)
    st = pd.read_csv(config.STATIONS_CSV)
    zfile = lambda r, y, d: os.path.join(config.CONTINUOUS, r.Network, r.Code, f"{r.Sensor}Z.D", f"{r.Network}.{r.Code}..{r.Sensor}Z.D.{y}.{d:03d}")
    active = {}
    for y, d in set(zip(df.time.dt.year, df.time.dt.dayofyear)):
        m = np.array([os.path.exists(zfile(r, y, d)) for r in st.itertuples()])
        active[(y, d)] = m if m.any() else np.ones(len(st), bool)
    df["nearest_sta_km"] = [round(float(_dist_km(st.Latitude[active[(t.year, t.dayofyear)]], st.Longitude[active[(t.year, t.dayofyear)]], la, lo).min()), 1)
                            for t, la, lo in zip(df.time, df.lat, df.lon)]
    # the study's one quality rule (user, 2026-10-13): the relocation's gate. quality_gate = config.HAENAM_QC (sumio.apply_qc
    # semantics); relocation_set = relocation/code/relocation_selection.py (gate, 5 km radius, same-second rule, depth;
    # = run_dl_hypodd.py --radius-region --min-depth 10). The tier columns stay as labels; nothing in the study reads them.
    q = config.HAENAM_QC
    df["quality_gate"] = ((df.erh < q["erh"]) & (df.erz < q["erz"]) & (df.gap < q["gap"]) & (df.num > q["num"]) & (df.rms < q["rms"]))
    sys.path.insert(0, str(haenam.ROOT / "relocation/code")); import relocation_selection as rsel   # the one definition
    gb = df.quality_gate & rsel.in_region(df.lat, df.lon)     # as catalog_from_sum: gate, region, same-second pair dropped, THEN depth
    sec = df.time.dt.floor("s").where(gb)
    first = gb & ~(sec.duplicated(keep="first") & gb)
    df["relocation_set"] = first & (df.depth >= rsel.MIN_DEPTH_KM)

    kma = kma_events()
    km = [nearest(t, kma, KMA_TOL_S) for t in df.time]
    df["KMA_match"] = [f"M{kma.loc[i, 'magnitude']:g}" if i is not None else "-" for i in km]
    df["KMA_dist_km"] = [round(float(_dist_km(kma.loc[i, "latitude"], kma.loc[i, "longitude"], la, lo)), 1)
                         if i is not None else np.nan for i, la, lo in zip(km, df.lat, df.lon)]
    if os.path.exists(TEMPLATES_CSV):                                     # GWANGYANG PORT: templates exist only after G7
        tp = pd.read_csv(TEMPLATES_CSV, dtype={"event_id": str})
    else:
        tp = pd.DataFrame(dict(event_id=pd.Series(dtype=str), origin_time_UTC=pd.Series(dtype=str)))
    tp["time"] = pd.to_datetime(tp.origin_time_UTC)
    tm = [nearest(t, tp, TPL_TOL_S) for t in df.time]
    df["template_match"] = [tp.loc[i, "event_id"] if i is not None else "-" for i in tm]
    df["unreported"] = df.KMA_match == "-"
    df["promote"] = df.tier.isin(["A", "B"]) & df.in_haenam_box & (df.template_match == "-")
    df["event_id"] = df.time.dt.strftime("%Y%m%d%H%M%S")

    out = df[["event_id", "time", "lat", "lon", "depth", "num", "gap", "rms", "erh", "erz", "qual",
              "quality_gate", "relocation_set", "tier",
              "dist_haenam_km", "in_haenam_box", "nearest_sta_km", "KMA_match", "KMA_dist_km",
              "template_match", "unreported", "promote"]].sort_values("time").reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"located events: {len(out)}  ({', '.join(os.path.basename(s) for s in sums)})")
    print(f"  quality gate (= the relocation's): {int(out.quality_gate.sum())}; relocation set (gate, 5 km radius, >= 10 km): {int(out.relocation_set.sum())}")
    box = out[out.in_haenam_box]
    print(f"in Haenam box: {len(box)}  (quality gate: {int(box.quality_gate.sum())}, template-matched: "
          f"{(box.template_match!='-').sum()}, KMA-matched: {(~box.unreported).sum()})")
    ntpl = len(tp[tp.time.between(out.time.min(), out.time.max())]) if len(out) else 0
    print(f"template (relocated) events inside the located time span: {ntpl}; recovered: "
          f"{out.template_match.nunique() - (1 if (out.template_match=='-').any() else 0)}")
    print(f"promote (A/B, in box, not in template catalog): {out.promote.sum()}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
