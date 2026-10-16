"""Stage DL-detected Haenam events for the canonical relocation (PocketQuake --augment).

WHY THIS ROUTE: the 195-event Haenam_joint(+JNU) relocation is the project's canonical relative frame
(metre-level bootstrap errors). New small events found by the DL pipeline should join THAT frame through
the SAME validated chain (PhaseNet+ per-event picks -> HYPOINVERSE -> xcorr with resume -> HypoDD ->
bootstrap) rather than through a second, independent relocation. PocketQuake already does exactly that
for new catalog events (`--augment`); it only needs (1) the events in a KMA-format catalog and (2) their
waveforms on disk in one of the cluster's per-event layouts. This script provides both from the
continuous archive — nothing is downloaded.

Layout written (STP per-event tree, which the mixed Haenam_joint cluster already enumerates):
    <cluster src_root>/stp_download/SAC/<event_id>/{HH,HG,EL}/<ts>.<NET>.<STA>.<CHAN>.sac
    ts = window start (origin - PRE_S) as YYYYMMDDHHMMSS; 500 s windows (the run's STP convention);
    plain SAC (reference = trace start, b = 0) — gather_event injects evla/evlo/stla/stlo/nz* itself.
    JU (200 Hz) stations are NOT staged here: they are added afterwards by relocation/jnu_augmentation/run_jnu_augment.sh
    (the station-augmentation step keeps them at native rate).
Catalog written: catalogs/haenam_augment_catalog.csv = the run's current catalog + the new events
    (KMA format, KST times; Magnitude = ML from the DL catalog if any, else 0.0).

Selection: rows of catalogs/haenam_dl_catalog.csv with promote == True (tier A/B, inside HAENAM_BOX,
not already in the template catalog). Override with --events / --min-tier.

Usage:
    python reloc/stage_for_augment.py                 # DRY RUN: list the events + counts, write nothing
    python reloc/stage_for_augment.py --apply         # cut SAC + write the augmented catalog
Then (user-run, in 15.PocketQuake):
    ./pocketquake.sh <DL_detection>/catalogs/haenam_augment_catalog.csv Haenam_joint --augment --dry-run
    ./pocketquake.sh <DL_detection>/catalogs/haenam_augment_catalog.csv Haenam_joint --augment --skip-download
    27.Haenam_sequence/JNU_augment/run_jnu_augment.sh run          # re-add the JNU stations (branch run)
"""
import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # the project registry, found upward from this file
import argparse
import glob
import os
import sys

import pandas as pd
from obspy import UTCDateTime, read

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, str(haenam.ROOT / 'detection/pipeline'))
import config  # noqa: E402

DL_CATALOG = haenam.path(None, 'dl_catalog')
KCR = "/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation"
CLUSTER_SRC = os.path.join(KCR, "haenam_joint_cluster")                  # Haenam_joint src_root
CURRENT_CATALOG = os.path.join(CLUSTER_SRC, "event_catalog", "event_catalog.csv")   # KMA format, KST
OUT_CATALOG = str(haenam.ROOT / 'DL_detection/catalogs/haenam_augment_catalog.csv')
STP_ROOT = os.path.join(CLUSTER_SRC, "stp_download", "SAC")
PRE_S, TOTAL_S = 30.0, 500.0
NETWORKS = ("KS", "KG")           # KG has no continuous data here -> skipped automatically
SENSOR_DIRS = {"HH": "HH", "HG": "HG", "EL": "EL"}


def day_files(net, sta, t):
    """All component day files of a station for the day containing t (any sensor)."""
    return sorted(glob.glob(f"{config.CONTINUOUS}/{net}/{sta}/*/*.D.{t.year}.{t.julday:03d}"))


def cut_event(eid, origin, stations, apply):
    t0, t1 = origin - PRE_S, origin - PRE_S + TOTAL_S
    ts = t0.strftime("%Y%m%d%H%M%S")
    n = 0
    for r in stations.itertuples():
        for f in day_files(r.Network, r.Code, origin):
            cha = os.path.basename(f).split(".")[3]
            sensor = cha[:2]
            if sensor not in SENSOR_DIRS:
                continue
            if not apply:
                n += 1
                continue
            try:
                st = read(f, starttime=t0 - 1, endtime=t1 + 1)
                if not st:
                    continue
                for tr in st:                       # NECIS 99.99999 Hz vs 100 Hz segments (see core.py)
                    if abs(tr.stats.sampling_rate - round(tr.stats.sampling_rate)) < 1e-3:
                        tr.stats.sampling_rate = float(round(tr.stats.sampling_rate))
                st.merge(method=1, fill_value=0)
                tr = st[0]
                tr.trim(t0, t1, pad=True, fill_value=0.0)
                if tr.stats.npts < 100:
                    continue
                d = os.path.join(STP_ROOT, eid, SENSOR_DIRS[sensor])
                os.makedirs(d, exist_ok=True)
                tr.write(os.path.join(d, f"{ts}.{r.Network}.{r.Code}.{cha}.sac"), format="SAC")
                n += 1
            except Exception as e:  # noqa: BLE001
                print(f"    {eid} {r.Network}.{r.Code}.{cha}: {type(e).__name__}: {e}")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write SAC + catalog (default: dry run)")
    ap.add_argument("--events", default=None, help="comma list of event_ids to stage instead of promote==True")
    ap.add_argument("--min-tier", default="B", choices=["A", "B"], help="minimum QC tier (default B)")
    a = ap.parse_args()

    if not os.path.exists(DL_CATALOG):
        sys.exit(f"no DL catalog at {DL_CATALOG} — run ./run_haenam.sh locate + catalog first")
    dl = pd.read_csv(DL_CATALOG, dtype={"event_id": str}, parse_dates=["time"])
    if a.events:
        sel = dl[dl.event_id.isin(a.events.split(","))]
    else:
        tiers = ["A"] if a.min_tier == "A" else ["A", "B"]
        sel = dl[dl.promote & dl.tier.isin(tiers)]
    cur = pd.read_csv(CURRENT_CATALOG)
    cur_ids = set((pd.to_datetime(cur[["Year", "Month", "Day", "Hour", "Minute", "Second"]])
                   - pd.Timedelta(hours=9)).dt.strftime("%Y%m%d%H%M%S"))
    sel = sel[~sel.event_id.isin(cur_ids)].sort_values("time").reset_index(drop=True)
    print(f"[stage] {len(sel)} DL events to stage (tier >= {a.min_tier}, in box, not in the run's catalog)")
    if sel.empty:
        return
    st = pd.read_csv(config.STATIONS_CSV)
    st = st[st.Network.isin(NETWORKS)]

    rows = []
    for r in sel.itertuples():
        o = UTCDateTime(r.time.to_pydatetime())
        exists = os.path.isdir(os.path.join(STP_ROOT, r.event_id))
        n = 0 if exists else cut_event(r.event_id, o, st, a.apply)
        print(f"  {r.event_id}  tier {r.tier}  {r.lat:.4f} {r.lon:.4f} {r.depth:.1f} km  N={int(r.num)}  "
              f"{'already staged' if exists else f'{n} channel files'}")
        kst = r.time + pd.Timedelta(hours=9)
        rows.append(dict(Year=kst.year, Month=kst.month, Day=kst.day, Hour=kst.hour, Minute=kst.minute,
                         Second=int(kst.second), Latitude=round(r.lat, 4), Longitude=round(r.lon, 4),
                         Depth=round(r.depth, 1), Magnitude=0.0))
    if a.apply:
        out = pd.concat([cur, pd.DataFrame(rows)], ignore_index=True)
        out["_t"] = pd.to_datetime(out[["Year", "Month", "Day", "Hour", "Minute", "Second"]])
        out = out.sort_values("_t").drop(columns="_t")
        out.to_csv(OUT_CATALOG, index=False)
        print(f"[stage] wrote {OUT_CATALOG}: {len(cur)} existing + {len(rows)} new = {len(out)} events")
        print(f"[stage] SAC staged under {STP_ROOT}/<event_id>/{{HH,HG,EL}}/")
        print("next (user-run):  cd /home/msseo/works/15.PocketQuake && ./pocketquake.sh "
              f"{OUT_CATALOG} Haenam_joint --augment --dry-run")
    else:
        print("[stage] DRY RUN — nothing written (add --apply)")


if __name__ == "__main__":
    main()
