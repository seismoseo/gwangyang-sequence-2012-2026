#!/usr/bin/env python3
"""Bit-exactness test of kcr_fast_xcorr against korea-cluster-relocation's own GPU batching, on real tasks of the R4 run
(same configuration: S on T, onset-window overrides, the run's interpolation cache). For the first N event pairs, every
(pair, phase, station, component) task is built and correlated both ways in identical batches; the (shift, cc) of every
task must be equal (==, not approximately). Also reports the time of both.

  pq-gpu python relocation/code/test_kcr_fast_xcorr.py [--pairs 60] [--batch 64]
"""
import argparse, os, sys, time
from glob import glob
from itertools import combinations
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "detection/pipeline"))
import haenam_dl_cluster_config as hcc   # noqa: E402  (puts the framework on sys.path)
from pipeline import config as kconfig   # noqa: E402
from pipeline.core import xcorr as X, sumio, evmap   # noqa: E402
import kcr_fast_xcorr as F               # noqa: E402

RUN = os.path.join(ROOT, "relocation/hypodd_dl_renewal_R4")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs", type=int, default=60); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--first-event", type=int, default=0, help="pairs among events[first : first + n] (n from --pairs)")
    ap.add_argument("--fast-first", action="store_true", help="run the fast path first (the second run finds warm traces)")
    a = ap.parse_args()
    import pandas as pd, torch
    sys.path.insert(0, HERE); import run_dl_hypodd as R, event_sac_export as ese
    import config as pc
    # the run's catalog and onset mapping (as run_dl_hypodd)
    parts = [ese.catalog_from_sum(os.path.join(pc.velmodel_dir("phasenet_plus", "kim1983"), f"HAENAM{y}.sum"), qc=pc.HAENAM_QC, radius_region=True)
             for y in range(2018, 2027)]
    cat = pd.concat(parts).sort_values("time").reset_index(drop=True); cat = cat[cat.depth >= 10].reset_index(drop=True)
    tm = R.template_matches(cat); inv = {t: d for d, (t, _) in tm.items()}; ids = set(cat.event_id.astype(str))
    onset = [inv[t] if t in inv else t for t in hcc.CANONICAL_ONSET_EVENTS_2020 + hcc.CANONICAL_ONSET_EVENTS_2026 if t in inv or t in ids]
    ms = inv.get(hcc.CANONICAL_MAINSHOCK)
    cfg = hcc.make_config(output_root=RUN, num_cores=24, onset_dl_ids=onset, mainshock_dl_id=ms, open_wdcc=True, s_comps=("T",))
    xc = dict(interp_hz=1000, bandpass=(5, 20), pre=0.5, post=0.5, margin=0.5, cc_threshold=0.7, p_comp="Z", s_comps=("N", "E"),
              shift_samp=500, slide_step=0.001)
    xc.update(cfg.xcorr); xc["bandpass"] = tuple(xc["bandpass"]); xc["s_comps"] = tuple(xc["s_comps"])
    common = kconfig.waveforms_dir(cfg)
    sumdf = sumio.read_sum(kconfig.sum_file(cfg, "kim1983")); dir_of = evmap.dir_of_cuspid(cfg)
    events, eid = [], {}
    for r in sumdf.itertuples():
        e = dir_of.get(int(r.id))
        if e is not None:
            events.append(e); eid[e] = int(r.id)
    stations = sorted({os.path.basename(f).split(".")[2] for e in events for f in glob(os.path.join(common, e, "*.sac"))})
    n = int((1 + (1 + 8 * a.pairs) ** 0.5) / 2) + 1
    pairs = list(combinations(events[a.first_event:a.first_event + n], 2))[:a.pairs]
    print(f"{len(pairs)} pairs among events {a.first_event}..{a.first_event + n - 1}; onset pairs: "
          f"{sum(1 for p in pairs if any(set(p) & set(k) for k in cfg.xcorr_pair_overrides))}")
    X._init_worker(common, stations, eid, xc, dict(cfg.xcorr_pair_overrides), None, None, "cctorch_gpu_batched",
                   os.path.join(RUN, "wf_interp_cache"))
    device = torch.device("cuda:0")
    sr = float(xc["interp_hz"]); nlag = int(xc["shift_samp"])
    slides = np.arange(-xc["margin"], xc["margin"], xc["slide_step"])
    orig_build, orig_proc = X._build_one_task, X._process_batch_safe
    specs = [(p, ph, hdr, sta, comp) for p in pairs for sta in stations
             for (ph, hdr, comp) in [("P", "a", "Z")] + [("S", "t0", c) for c in xc["s_comps"]]]
    t0 = time.time(); nw = 0                                 # warm-up: load every trace, so both paths time the batching only
    for (p, ph, hdr, sta, comp) in specs:
        pre, post, fmin, fmax = X._window(p)
        for e in p:
            try:
                X._full_trace(e, sta, comp, fmin, fmax); nw += 1
            except Exception:
                pass
    print(f"warm-up: {nw} trace loads in {time.time() - t0:.1f} s", flush=True)
    res = {}
    runs = (("framework", orig_build, orig_proc), ("fast", F.build_one_task, F.process_batch_safe))
    for name, build, proc in (runs[::-1] if a.fast_first else runs):
        stats = {"oom_retries": 0, "cpu_fallback_tasks": 0, "batches": 0, "tasks": 0}
        out = {}; batch = []; nref = None; t0 = time.time()
        torch.cuda.synchronize()
        for (p, ph, hdr, sta, comp) in specs:
            pre, post, fmin, fmax = X._window(p)
            try:
                b = build(p[0], p[1], sta, comp, hdr, pre, post, fmin, fmax, slides, sr)
            except Exception:
                b = None
            if b is None:
                continue
            ref, cands, valid, n_ref = b
            if nref is None:
                nref = n_ref
            if n_ref != nref or len(batch) >= a.batch:
                out.update(proc(batch, nlag, slides, sr, device, stats)); batch, nref = [], n_ref
            batch.append(dict(key=(p, ph, sta, comp), ref=ref, cands=cands, valid=valid, n_ref=n_ref))
        out.update(proc(batch, nlag, slides, sr, device, stats)); torch.cuda.synchronize()
        res[name] = (out, time.time() - t0)
        print(f"{name}: {len(out)} tasks in {res[name][1]:.1f} s ({stats['batches']} batches, OOM retries {stats['oom_retries']})", flush=True)
    A, B = res["framework"][0], res["fast"][0]
    same_keys = set(A) == set(B)
    diff = [k for k in A if k in B and A[k] != B[k]]
    print(f"tasks: framework {len(A)}, fast {len(B)}, same task set: {same_keys}; tasks with any difference in (shift, cc): {len(diff)}")
    if diff:
        for k in diff[:5]:
            print("  DIFF", k, A[k], B[k])
    print("BIT-EXACT" if same_keys and not diff else "NOT EQUAL", f"| speed-up {res['framework'][1] / max(res['fast'][1], 1e-9):.1f}x")


if __name__ == "__main__":
    main()
