"""Well-constrained HYPOINVERSE events only: maps, sections and TIME SERIES for a QC gate.

Gate = config.HAENAM_QC (gap < 180 deg, N >= 8, RMS < 1 s, ERH < 5 km, ERZ < 5 km), optionally tightened.
Functions are importable (notebook 02 uses them); the CLI writes PNGs into source/events/20200503T1307/slip/figs/:
    qc_map_<tag>.png, qc_sections_<tag>.png, qc_timeseries_<tag>.png (full span + sequence window)

Run:  python analysis/map_qc_events.py                                  # gate
      python analysis/map_qc_events.py --erh 1 --erz 2 --rms 0.5 --tag tight
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline")); import config  # noqa: E402
from sumio import read_sum, apply_qc  # noqa: E402
import pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # registry paths

EPI = (34.660, 126.405); BOX = config.HAENAM_BOX
MAP_REGION = [125.6, 127.2, 33.95, 35.35]; ZOOM_REGION = [126.30, 126.52, 34.58, 34.74]
SEQ_START, SEQ_END = pd.Timestamp("2026-08-14"), pd.Timestamp("2026-09-09")
DL_ONSET = pd.Timestamp("2026-08-14 05:34:13")               # first DL-located event at the source (3.2 h before the KMA M2.1)
KMA_ONSET = pd.Timestamp("2026-08-14 08:43:38")              # first KMA-reported event of the sequence (M2.1)
MAJOR_ML = 2.0                                               # KMA events marked as verticals in the sequence panel
FULLNET = (pd.Timestamp("2026-05-10"), pd.Timestamp("2026-09-03"))     # KS archive present (DOY 130-245)
TIGHT = dict(erh=1.0, erz=2.0, rms=0.5)


def gate_text(qc):
    return f"gap<{qc['gap']:g}, N>={qc['num']+1}, RMS<{qc['rms']:g} s, ERH<{qc['erh']:g}, ERZ<{qc['erz']:g} km"


def select(qc=None, year=2026, model="phasenet_plus", **tighten):
    """Located events passing the gate (dict like config.HAENAM_QC, keys overridable via **tighten).
    Returns (all_passing, in_box) with dist_km / core columns."""
    q = dict(qc or config.HAENAM_QC); q.update(tighten)
    df = read_sum(os.path.join(config.velmodel_dir(model, "kim1983"), f"{config.REGION_NAME}{year}.sum"))
    df = apply_qc(df, qc=q)
    df["dist_km"] = 111.19 * np.hypot(df.lat - EPI[0], (df.lon - EPI[1]) * np.cos(np.radians(EPI[0])))
    df["core"] = (df.dist_km <= 3) & df.depth.between(15, 25)
    box = df[df.lat.between(BOX[0], BOX[1]) & df.lon.between(BOX[2], BOX[3])].copy()
    return df, box, q


def templates(year=2026):
    tp = pd.read_csv(haenam.path(None, "tm_templates_195"), dtype={"event_id": str})
    tp["time"] = pd.to_datetime(tp.origin_time_UTC)
    return tp[tp.time.dt.year == year]


KMA_CSV = "/home/msseo/works/07.SeismoStats/catalog/kma_entire_catalog_260309.csv"          # to 2026-03-09, KST
KMA_2026_CSV = "/home/msseo/works/15.PocketQuake/catalogs/Haenam_joint_catalog_2026sep03.csv"  # KMA list used for the relocation, KST


def kma_events(year=2026, box=BOX):
    """KMA-reported events in the box with magnitudes (UTC): the entire KMA catalog plus the 2026 Haenam
    list used for the canonical relocation (both KST -> UTC), de-duplicated within 2 s."""
    k = pd.read_csv(KMA_CSV, parse_dates=["time"], encoding="utf-8-sig")
    k = pd.DataFrame(dict(time=k.time - pd.Timedelta(hours=9), latitude=k.latitude, longitude=k.longitude,
                          depth_km=pd.to_numeric(k.depth, errors="coerce"), magnitude=pd.to_numeric(k.magnitude, errors="coerce")))
    if os.path.exists(KMA_2026_CSV):
        k2 = pd.read_csv(KMA_2026_CSV)
        k2 = pd.DataFrame(dict(time=pd.to_datetime(k2[["Year", "Month", "Day", "Hour", "Minute", "Second"]]) - pd.Timedelta(hours=9),
                               latitude=k2.Latitude, longitude=k2.Longitude, depth_km=k2.Depth, magnitude=k2.Magnitude))
        k = pd.concat([k, k2], ignore_index=True)
    k = k[(k.time.dt.year == year) & k.latitude.between(box[0], box[1]) & k.longitude.between(box[2], box[3])].sort_values("time")
    gaps = np.diff(k.time.values.astype("datetime64[ns]")) / np.timedelta64(1, "s")
    keep = np.r_[True, gaps > 2]
    return k[keep].reset_index(drop=True)


def _helvetica():
    sys.path.insert(0, "/home/msseo/works/24.Slip_inversion/pipeline/notebooks")
    try:
        from nbfont import use_helvetica; use_helvetica()
    except Exception:  # noqa: BLE001
        pass


def plot_maps(df, box, q, tag, figdir, show=False):
    import pygmt
    sta = pd.read_csv(config.STATIONS_CSV); tpy = templates()
    pygmt.config(FORMAT_GEO_MAP="ddd.xx", MAP_FRAME_TYPE="plain", FONT="10p,Helvetica")
    fig = pygmt.Figure()
    fig.basemap(region=MAP_REGION, projection="M9c", frame=["af", f"WSen+tQC-passed ({len(df)})"])
    fig.coast(shorelines="0.5p,gray40", land="gray95", water="white")
    pygmt.makecpt(cmap="viridis", series=[0, 25])
    fig.plot(x=df.lon, y=df.lat, fill=df.depth, cmap=True, style="c0.12c", pen="0.2p,gray30")
    fig.plot(x=sta.Longitude, y=sta.Latitude, style="t0.2c", fill="firebrick", pen="0.3p,black")
    fig.plot(x=[BOX[2], BOX[3], BOX[3], BOX[2], BOX[2]], y=[BOX[0], BOX[0], BOX[1], BOX[1], BOX[0]], pen="0.8p,black,--")
    fig.basemap(map_scale="jBL+w20k+o0.5c/0.5c")
    fig.shift_origin(xshift="10c")
    fig.basemap(region=ZOOM_REGION, projection="M9c", frame=["af", f"wSEn+tHaenam box ({len(box)})"])
    fig.coast(shorelines="0.5p,gray40", land="gray97", water="white")
    fig.plot(x=box.lon, y=box.lat, fill=box.depth, cmap=True, style="c0.14c", pen="0.2p,gray30")
    km = kma_events()                                   # GMT's auto-legend hangs on variable-size symbols: no label here
    fig.plot(x=km.longitude, y=km.latitude, size=0.06 + 0.09 * km.magnitude.clip(0, 4), style="cc", fill="white", pen="0.5p,black")
    s3 = sta[sta.Longitude.between(*ZOOM_REGION[:2]) & sta.Latitude.between(*ZOOM_REGION[2:])]
    fig.plot(x=s3.Longitude, y=s3.Latitude, style="t0.3c", fill="firebrick", pen="0.3p,black")
    fig.text(x=s3.Longitude, y=s3.Latitude, text=s3.Code, offset="0.25c/0.2c", font="7p,Helvetica,black")
    fig.colorbar(frame=["af", "+lDepth (km)"], position="JMR+o1.1c/0c+w5.5c")
    fig.text(position="TL", offset="0.25c/-0.25c", text=f"white circles: KMA events ({len(km)}), size = ML", font="8p,Helvetica,black", fill="white", pen="0.5p,black", justify="TL")
    fig.basemap(map_scale="jBL+w5k+o0.5c/0.5c")
    out = os.path.join(figdir, f"qc_map_{tag}.png"); fig.savefig(out, dpi=200)
    if show:
        fig.show()
    return out


def plot_sections(box, q, tag, figdir):
    import matplotlib.pyplot as plt
    tpy = templates(); cosl = np.cos(np.radians(EPI[0]))
    x = (box.lon - EPI[1]) * 111.19 * cosl; y = (box.lat - EPI[0]) * 111.19
    tx = (tpy.longitude - EPI[1]) * 111.19 * cosl; ty = (tpy.latitude - EPI[0]) * 111.19
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].scatter(x, y, s=7, c="steelblue", label=f"QC-passed in box ({len(box)})"); ax[0].scatter(tx, ty, s=4, c="k", label=f"relocated ({len(tpy)})")
    ax[0].set_xlabel("East (km)"); ax[0].set_ylabel("North (km)"); ax[0].set_title("Map view"); ax[0].legend(fontsize=7, framealpha=1)
    ax[1].scatter(x, box.depth, s=7, c="steelblue"); ax[1].scatter(tx, tpy.depth_km, s=4, c="k"); ax[1].set_xlabel("East (km)"); ax[1].set_ylabel("Depth (km)"); ax[1].set_title("E-W section"); ax[1].invert_yaxis()
    ax[2].scatter(y, box.depth, s=7, c="steelblue"); ax[2].scatter(ty, tpy.depth_km, s=4, c="k"); ax[2].set_xlabel("North (km)"); ax[2].set_ylabel("Depth (km)"); ax[2].set_title("N-S section"); ax[2].invert_yaxis()
    for a_ in ax: a_.set_aspect("equal")
    fig.suptitle(f"HYPOINVERSE, QC gate: {gate_text(q)}", fontsize=10); plt.tight_layout()
    out = os.path.join(figdir, f"qc_sections_{tag}.png"); plt.savefig(out, dpi=160); return fig, out


def plot_timeseries(box_gate, box_tight, q_gate, q_tight, figdir, tag="gate_vs_tight"):
    """Daily counts + cumulative curves of in-box events for the gate and the tight gate, over the full
    2026 span (top row) and the sequence window (bottom row); canonical relocated events for reference."""
    import matplotlib.pyplot as plt
    tpy = kma_events()                                   # KMA-reported events with magnitudes (UTC)
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    for row, (t0, t1, lab) in enumerate([(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-09-09"), "full 2026 span"),
                                         (SEQ_START - pd.Timedelta(days=4), SEQ_END, "sequence window")]):
        sel = lambda d: d[(d.time >= t0) & (d.time < t1)]
        g, tt, rel = sel(box_gate), sel(box_tight), sel(tpy.rename(columns={"time": "time"}))
        freq, width = ("D", pd.Timedelta("1D")) if row == 0 else ("6h", pd.Timedelta("6h"))
        for d, col, lab2, z in [(g, "lightsteelblue", f"gate ({len(g)})", 1), (tt, "steelblue", f"tight ({len(tt)})", 2)]:
            c = d.groupby(d.time.dt.floor(freq)).size()
            ax[row, 0].bar(c.index, c.values, width=width, color=col, label=lab2, zorder=z, align="edge")
        ax[row, 0].set_ylabel(f"Events per {'day' if row == 0 else '6 h'}"); ax[row, 0].set_title(f"In-box events, {lab}")
        axm = ax[row, 0].twinx()                          # KMA magnitudes as stems on the right axis
        axm.vlines(rel.time, 0, rel.magnitude, color="firebrick", lw=1.0, alpha=0.8, zorder=4)
        axm.plot(rel.time, rel.magnitude, "o", ms=3, color="firebrick", zorder=5, label=f"KMA ML ({len(rel)})")
        for r in rel[rel.magnitude >= (2.5 if row == 0 else MAJOR_ML)].itertuples():
            axm.annotate(f"{r.magnitude:.1f}", (r.time, r.magnitude), xytext=(2, 2), textcoords="offset points", fontsize=7, color="firebrick")
        axm.set_ylim(0, 4.0); axm.set_ylabel("KMA ML", color="firebrick"); axm.tick_params(axis="y", colors="firebrick")
        h1, l1 = ax[row, 0].get_legend_handles_labels(); h2, l2 = axm.get_legend_handles_labels()
        ax[row, 0].legend(h1 + h2, l1 + l2, framealpha=1, fontsize=8, loc="upper right")
        for d, col, lab2 in [(g, "lightsteelblue", "gate"), (tt, "steelblue", "tight"), (rel, "firebrick", "KMA")]:
            t = np.sort(d.time.values); ax[row, 1].step(t, np.arange(1, len(t) + 1), where="post", color=col, lw=1.5, label=f"{lab2} ({len(t)})")
        big = rel[rel.magnitude >= MAJOR_ML]
        ax[row, 1].plot(big.time, [int((rel.time <= t).sum()) for t in big.time], "o", color="firebrick", ms=5, zorder=5)
        if row == 1:                                  # sequence window: major events as dashed verticals
            ymax = ax[row, 1].get_ylim()[1] if ax[row, 1].get_ylim()[1] > 10 else 500
            ax[row, 1].axvline(DL_ONSET, color="steelblue", ls="--", lw=1.0)
            ax[row, 1].text(DL_ONSET, 0.98, " DL onset 05:34", rotation=90, va="top", ha="right", fontsize=7, color="steelblue", transform=ax[row, 1].get_xaxis_transform())
            for k, r in enumerate(big.itertuples()):
                ax[row, 1].axvline(r.time, color="firebrick", ls="--", lw=0.9, alpha=0.8)
                ax[row, 1].text(r.time, 0.98 - 0.10 * (k % 2), f" M{r.magnitude:.1f}", rotation=90, va="top", ha="right", fontsize=7, color="firebrick", transform=ax[row, 1].get_xaxis_transform())
        ax[row, 1].set_ylabel("Cumulative count"); ax[row, 1].set_title(f"Cumulative, {lab}")
        ax[row, 1].legend(loc="upper left" if row == 0 else "center right", framealpha=1, fontsize=8)
        for a_ in ax[row]:
            a_.set_xlim(t0, t1)
            if row == 0:
                a_.axvspan(FULLNET[0], FULLNET[1], color="0.93", zorder=0)
            a_.axvline(KMA_ONSET, color="red", ls="--", lw=0.8)
    import matplotlib.dates as mdates
    for a_ in ax[1]:
        a_.xaxis.set_major_locator(mdates.DayLocator(interval=4)); a_.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    for a_ in ax[0]:
        a_.xaxis.set_major_locator(mdates.MonthLocator()); a_.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax[1, 0].set_xlabel("Date (UTC), 2026"); ax[1, 1].set_xlabel("Date (UTC), 2026")
    fig.suptitle(f"gate: {gate_text(q_gate)}   |   tight: {gate_text(q_tight)}\n"
                 f"gray: KS network present;  red dashed: KMA onset (M2.1, 08:43 UTC);  blue dashed: DL onset (05:34 UTC);  "
                 f"red stems/dots: KMA events, ML >= {MAJOR_ML:g} marked in the sequence panels", fontsize=9)
    plt.tight_layout()
    out = os.path.join(figdir, f"qc_timeseries_{tag}.png"); plt.savefig(out, dpi=160); return fig, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026); ap.add_argument("--model", default="phasenet_plus")
    ap.add_argument("--erh", type=float); ap.add_argument("--erz", type=float); ap.add_argument("--rms", type=float); ap.add_argument("--gap", type=float)
    ap.add_argument("--num", type=int, help="minimum picks (N >= num)")
    ap.add_argument("--tag", default="gate"); ap.add_argument("--figdir", default=str(haenam.ROOT / 'analysis/seismicity/figs'))
    a = ap.parse_args()
    tighten = {k: getattr(a, k) for k in ("erh", "erz", "rms", "gap") if getattr(a, k) is not None}
    if a.num is not None:
        tighten["num"] = a.num - 1
    import matplotlib; matplotlib.use("Agg"); _helvetica()
    os.makedirs(a.figdir, exist_ok=True)
    df, box, q = select(year=a.year, model=a.model, **tighten)
    core = box[box.core]
    print(f"{len(df)} pass [{gate_text(q)}]; in box {len(box)}; core {len(core)}, depth 5-95% {core.depth.quantile(.05):.1f}-{core.depth.quantile(.95):.1f} km")
    print("->", plot_maps(df, box, q, a.tag, a.figdir)); print("->", plot_sections(box, q, a.tag, a.figdir)[1])
    dg, bg, qg = select(year=a.year, model=a.model); dt_, bt, qt = select(year=a.year, model=a.model, **TIGHT)
    print("->", plot_timeseries(bg, bt, qg, qt, a.figdir)[1])


if __name__ == "__main__":
    main()
