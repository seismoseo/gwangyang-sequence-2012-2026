#!/usr/bin/env python
"""GWANGYANG G8 (port of Imsil reloc/run_fmf_hypodd.py, decision D22): joint HypoDD relocation of the template-matching
detections with the focused 25 m reference events, with S on T.

Same framework, settings and stages as relocation/code/run_dl_hypodd.py (korea-cluster-relocation from stage "hypoinverse":
hyp1.40 on the SAC-header picks -> ph2dt -> dt.ct -> rereference -> GPU xcorr (P on Z, S on T) -> dt.cc with the 7-set
schedule, open WDCC, adaptive damping -> 1000-replica bootstrap -> final_locations.csv). Two differences, as in Imsil D22:
  * events: the reference events (real PhaseNet+ picks) + every new detection (picks inherited from its template;
    relocation/code/export_fmf_detections.py --apply writes the run root first);
  * catalog differential times only among the references: ph2dt runs as usual, then again on a copy of the phase file
    that holds only the references, and that dt.ct replaces the full one (kept as dt.ct.all_events). The detections
    enter the inversion through dt.cc alone, so their inherited picks cannot pin them to their templates.
Differences from Imsil: registry paths; references read from every year's HYPOINVERSE .sum with the G5 radius selection;
S on T (make_config s_comps=("T",)); the onset window of G5 for the 2021 ML 2.2 when it is a reference (decision G-D18;
Imsil used none); optional --fast-xcorr as in G5; KMA ML from registry/events.csv.
Output: <run_root>/final_locations.csv and <run_root>/gwangyang_fmf_relocated.csv (+ kind, template, FMF CC, origin_time_reloc).
Run (GPU env, never with detection or template matching on the GPU):
    set -a; . /home/msseo/works/15.PocketQuake/.env; set +a
    /home/msseo/miniforge3/envs/pq-gpu/bin/python relocation/code/run_fmf_hypodd.py [--run-root relocation/hypodd_fmf_2012-2026] [--n-boot 1000] [--fast-xcorr]
"""
import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # the project registry, found upward from this file
import argparse
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
from obspy import UTCDateTime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, str(haenam.ROOT / "detection/pipeline"))
import config as pipe_config                                   # noqa: E402  (DL pipeline config)
import event_sac_export as ese                                 # noqa: E402
import haenam_dl_cluster_config as hcc                         # noqa: E402  (also puts KCR on sys.path)
from add_reloc_origin_times import reloc_times                 # noqa: E402

KMA_CSV = str(haenam.ROOT / "registry/events.csv")             # the 11 KMA events (origin_time_utc, ML)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default=str(haenam.ROOT / "relocation/hypodd_fmf_2012-2026"))
    ap.add_argument("--model", default="phasenet_plus")
    ap.add_argument("--velmodel", default="kim1983")
    ap.add_argument("--stage-from", default="hypoinverse", choices=["hypoinverse", "ph2dt", "dtct", "rereference", "xcorr", "dtcc"])
    ap.add_argument("--cores", type=int, default=24)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--blocked-pairs", type=int, default=45)
    ap.add_argument("--xcorr-chunk", type=int, default=4000)
    ap.add_argument("--xcorr-resume", action="store_true")
    ap.add_argument("--fast-xcorr", action="store_true", help="GPU candidate matrices (kcr_fast_xcorr.py), as G5")
    ap.add_argument("--bootstrap-only", action="store_true")
    ap.add_argument("--out-csv", default=None, help="final table (default <run_root>/gwangyang_fmf_relocated.csv)")
    a = ap.parse_args()
    ese.ROTATE_T = True
    if a.fast_xcorr:
        import kcr_fast_xcorr; kcr_fast_xcorr.install(prefetch=True)
    out_csv = a.out_csv or os.path.join(a.run_root, "gwangyang_fmf_relocated.csv")
    inp = pd.read_csv(os.path.join(a.run_root, "fmf_relocation_input.csv"), dtype={"event_id": str, "template_event_id": str})
    inp["template_event_id"] = inp.template_event_id.str.replace(r"\.0$", "", regex=True)   # written as float because reference rows are empty
    inp["template_cuspid"] = inp.template_cuspid.astype("Int64")
    ref_ids = set(inp.event_id[inp.kind == "reference"])

    # 1. catalog: the references from the HYPOINVERSE .sum of their years (same QC and radius selection as G5) + the detections
    #    at their template position
    det = inp[inp.kind == "detection"].copy()
    det["time"] = pd.to_datetime(det.origin); det["origin_utc"] = det.time.apply(lambda t: UTCDateTime(t.to_pydatetime()))
    years = sorted({int(e[:4]) for e in ref_ids})
    parts = []
    for y in years:
        sum_path = os.path.join(pipe_config.velmodel_dir(a.model, a.velmodel), f"{pipe_config.REGION_NAME}{y}.sum")
        parts.append(ese.catalog_from_sum(sum_path, qc=pipe_config.HAENAM_QC, radius_region=True))
    dl = pd.concat(parts)
    refs = dl[dl.event_id.isin(ref_ids)].copy()
    if len(refs) != len(ref_ids):
        raise SystemExit(f"{len(ref_ids) - len(refs)} reference events not found in the .sum files of {years}")
    for c in ("num", "gap", "rms", "erh", "erz"): det[c] = np.nan
    cols = ["event_id", "origin_utc", "time", "lat", "lon", "depth", "num", "gap", "rms", "erh", "erz"]
    cat = pd.concat([refs[cols], det[cols]]).sort_values("time").reset_index(drop=True)
    kma = pd.read_csv(KMA_CSV, parse_dates=["origin_time_utc"])
    mags = {}
    for r in refs.itertuples():
        dt = (kma.origin_time_utc - r.time).abs(); i = dt.idxmin()
        if dt[i].total_seconds() <= 6.0: mags[r.event_id] = float(kma.loc[i, "ML"])
    # onset window of G5 (decision G-D18): the 2021 ML 2.2, found by its KMA UTC time within 3 s among the references
    t_ms = pd.Timestamp(f"{hcc.CANONICAL_MAINSHOCK[:4]}-{hcc.CANONICAL_MAINSHOCK[4:6]}-{hcc.CANONICAL_MAINSHOCK[6:8]} "
                        f"{hcc.CANONICAL_MAINSHOCK[8:10]}:{hcc.CANONICAL_MAINSHOCK[10:12]}:{hcc.CANONICAL_MAINSHOCK[12:14]}")
    dms = (refs.time - t_ms).abs()
    ms_id = str(refs.event_id[dms.idxmin()]) if len(dms) and dms.min() <= pd.Timedelta(seconds=3) else None
    print(f"[catalog] {len(refs)} references + {len(det)} detections = {len(cat)} events -> {len(cat) * (len(cat) - 1) // 2:,} xcorr pairs")
    print(f"[onset] {'onset window for the 2021 ML 2.2 = ' + ms_id if ms_id else 'the 2021 ML 2.2 is not a reference: no onset window'}")
    cfg = hcc.make_config(output_root=a.run_root, num_cores=a.cores, onset_dl_ids=[ms_id] if ms_id else [], mainshock_dl_id=ms_id,
                          open_wdcc=True, s_comps=("T",))
    print(f"[xcorr] P on Z, S on {'+'.join(cfg.xcorr['s_comps'])}")
    ese.write_framework_catalog(cat, cfg.event_catalog_csv, magnitudes=mags)

    # 2. manifest (all events; SAC bundles were written by export_fmf_detections.py)
    from pipeline import config as kcr_config
    from pipeline.core import evmap
    wf_dir = kcr_config.waveforms_dir(cfg)
    missing = [e for e in cat.event_id if not os.path.isdir(os.path.join(wf_dir, e))]
    if missing:
        raise SystemExit(f"{len(missing)} events have no SAC bundle under {wf_dir} - run relocation/code/export_fmf_detections.py --apply first")
    mp = evmap.pin_manifest(cfg, new_event_ids=cat.event_id.astype(str))
    man = pd.read_csv(mp, dtype={"event_id": str})
    ref_cuspids = set(200000 + man.event_idx[man.event_id.isin(ref_ids)])
    print(f"[manifest] {mp}: {len(man)} events; {len(ref_cuspids)} reference cuspids")

    # 3. framework stages, with catalog differential times among the references only
    from pipeline.core import pipeline as kcr_pipeline, hypodd, xcorr as kcr_xcorr
    from pipeline import viz
    _run_ph2dt = hypodd.run_ph2dt

    def run_ph2dt_reference_dtct(cfg_):
        d = _run_ph2dt(cfg_)                                   # event.dat / event.sel / dt.ct for every event
        sub = os.path.join(d, "reference_only"); os.makedirs(sub, exist_ok=True)
        pha = f"{cfg_.region}_mod.pha"; keep = False; n_ev = 0
        with open(os.path.join(d, pha)) as fi, open(os.path.join(sub, pha), "w") as fo:
            for line in fi:
                if line.startswith("#"):
                    keep = int(line.split()[-1]) in ref_cuspids; n_ev += keep
                if keep: fo.write(line)
        for f in ("station.dat", "ph2dt.inp"): shutil.copy(os.path.join(d, f), sub)
        subprocess.run(["ph2dt", "ph2dt.inp"], cwd=sub, check=True, stdout=subprocess.DEVNULL)
        shutil.copy(os.path.join(d, "dt.ct"), os.path.join(d, "dt.ct.all_events"))
        shutil.copy(os.path.join(sub, "dt.ct"), os.path.join(d, "dt.ct"))
        n_pairs = sum(1 for ln in open(os.path.join(d, "dt.ct")) if ln.startswith("#"))
        print(f"[ph2dt] dt.ct restricted to the {n_ev} reference events ({n_pairs} event pairs); full file kept as dt.ct.all_events")
        return d
    hypodd.run_ph2dt = run_ph2dt_reference_dtct
    if a.xcorr_chunk:
        os.environ["XCORR_PAIR_CHUNK"] = str(a.xcorr_chunk)
    if a.blocked_pairs:
        # Cache-friendly pair order, copied from run_dl_hypodd.py (results are per pair; the order has no effect on dt.cc)
        B = a.blocked_pairs

        def blocked_combinations(events, r=2):
            ev = list(events)
            blocks = [ev[k:k + B] for k in range(0, len(ev), B)]
            for ia, ba in enumerate(blocks):
                for ib in range(ia, len(blocks)):
                    bb = blocks[ib]
                    for i, e1 in enumerate(ba):
                        for j, e2 in enumerate(bb):
                            if ib > ia or j > i:
                                yield (e1, e2)
        kcr_xcorr.combinations = blocked_combinations
        print(f"[xcorr-order] blocked pair order, blocks of {B} events; chunk={os.environ.get('XCORR_PAIR_CHUNK', 'default')}")
    if not a.bootstrap_only:
        res = kcr_pipeline.run_cluster(cfg, stage_from=a.stage_from, through="dtcc", velmodels=("kim1983", "kim2011"), arc_velmodel="kim1983",
                                       cores=a.cores, xcorr_resume=a.xcorr_resume)
        print("[run_fmf_hypodd] framework:", {k: v for k, v in res.items() if k != "waveforms"})

    # 4. bootstrap + final table
    if a.n_boot:
        hypodd.bootstrap_relocation(cfg, "dtcc", n=a.n_boot, cores=None)
    tab = viz.location_table(cfg, branch="dtcc", save=True)
    tab["event_id"] = tab.event_id.astype(str)
    meta = inp.set_index("event_id")[["kind", "template_cuspid", "template_event_id", "fmf_cc", "n_present", "n_P", "n_S"]]
    tab = tab.merge(meta, left_on="event_id", right_index=True, how="left")
    tab["KMA_ML"] = tab.event_id.map(mags)
    times = reloc_times(os.path.join(kcr_config.dtcc_dir(cfg), "hypoDD.reloc"), mp)
    tab = tab.merge(times, on="event_id", how="left")
    tab.to_csv(out_csv, index=False)
    for k in ("reference", "detection"):
        g = tab[tab.kind == k]
        print(f"[final] {k}: {len(g)} relocated; under-constrained {int(g.under_constrained.sum())}; cc links median {g.cc_links.median():.0f}; ct links median {g.ct_links.median():.0f}")
    print(f"[final] {len(tab)} events -> {out_csv}")


if __name__ == "__main__":
    main()
