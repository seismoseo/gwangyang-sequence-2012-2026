#!/usr/bin/env python3
"""Plain progress of the R4 bootstrap (korea-cluster-relocation hypodd.bootstrap_relocation): every replica runs in its own
temporary folder /tmp/boot_haenam_dl_dtcc_*, removed when it ends. This watcher lists those folders every 30 s, keeps every
name it has seen (state file), and every --every seconds prints one line: replicas finished (seen and gone) and running, of
the total, the rate over the last 30 minutes and the clock-time estimate of the end. Before the first replica it reports the
calibration probe (/tmp/bootprobe_haenam_dl_*).

  python tools/r4_boot_progress.py [--n 1000] [--every 1200]
"""
import argparse, glob, json, os, time
from datetime import datetime, timedelta

STATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs/r4_boot_progress_state.json")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--every", type=int, default=1200)
    a = ap.parse_args()
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"seen": {}, "gone": {}}
    last = 0.0
    while True:
        now = time.time()
        cur = set(os.path.basename(p) for p in glob.glob("/tmp/boot_haenam_dl_dtcc_*"))
        for d in cur:
            st["seen"].setdefault(d, now)
        for d in list(st["seen"]):
            if d not in cur and d not in st["gone"]:
                st["gone"][d] = now
        json.dump(st, open(STATE, "w"))
        if now - last >= a.every:
            last = now
            done = len(st["gone"]); run = len(cur); t = datetime.now()
            if not st["seen"]:
                probe = glob.glob("/tmp/bootprobe_haenam_dl_*")
                print(f"{t:%H:%M} R4 bootstrap: calibrating the damping on the un-resampled set ({'running' if probe else 'not started'}); "
                      f"no replica started yet", flush=True)
            else:
                win = min(1800.0, now - min(st["seen"].values()))     # the last 30 min, or since the first replica began
                recent = round(sum(1 for v in st["gone"].values() if now - v <= win) * 3600 / max(win, 1.0))
                eta = f"; ends ~{t + timedelta(hours=(a.n - done) / recent):%a %H:%M}" if recent else ""
                dur = [v - st["seen"][k] for k, v in st["gone"].items() if now - v <= win]
                slow = f"; slowest of these {max(dur):.0f} s (limit 1069 s)" if dur else ""
                print(f"{t:%H:%M} R4 bootstrap: {done:,} of {a.n:,} replicas finished ({100 * done / a.n:.0f}%), {run} running; "
                      f"{recent:,} per hour over the last {win / 60:.0f} min{eta}{slow}", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
