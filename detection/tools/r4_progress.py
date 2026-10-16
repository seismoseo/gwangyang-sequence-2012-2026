#!/usr/bin/env python3
"""One plain progress line for the R4 relocation run: the current stage, pairs cross-correlated (of all), the rate over
the last 30 minutes, and clock-time estimates for the end of the cross-correlation and of the whole run (then dt.cc
HypoDD ~1 h and the 1000-replica bootstrap ~5 h, as in the old run).

  python tools/r4_progress.py
"""
import glob, os, re, time
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(os.path.dirname(HERE), "relocation/hypodd_dl_renewal_R4")
LOG = os.path.join(HERE, "logs/r4_reloc_fast2.log")
TOTAL = 331705
DTCC_H, BOOT_H = 1.0, 5.0


def main():
    now = time.time()
    log = open(LOG, errors="replace").read() if os.path.exists(LOG) else ""
    files = glob.glob(os.path.join(RUN, "2.HypoDD/02.dt.cc/dt.cc_S/dt.cc_S_*"))
    done = len(files)
    stage = ("finished" if "[final]" in log else "bootstrap (1000 replicas)" if "dtcc[" in log
             else "dt.cc relocation (HypoDD)" if "[haenam_dl] xcorr:" in log else "cross-correlation")
    recent = sum(1 for f in files if now - os.path.getmtime(f) <= 1800)
    rate = recent * 2                                         # pairs per hour over the last 30 min
    m = re.findall(r"chunk (\d+)/(\d+)", log)
    chunk = f"chunk {m[-1][0]} of {m[-1][1]}" if m else ""
    t = datetime.now()
    if stage == "cross-correlation" and rate > 0:
        xend = t + timedelta(hours=(TOTAL - done) / rate)
        rend = xend + timedelta(hours=DTCC_H + BOOT_H)
        eta = f"cross-correlation ends ~{xend:%H:%M}, R4 ends ~{rend:%a %H:%M}"
    elif stage.startswith("dt.cc"):
        eta = f"R4 ends ~{(t + timedelta(hours=DTCC_H + BOOT_H)):%a %H:%M} (dt.cc ~1 h, bootstrap ~5 h)"
    elif stage.startswith("bootstrap"):
        eta = f"R4 ends ~{(t + timedelta(hours=BOOT_H)):%a %H:%M} at the latest (bootstrap ~5 h)"
    else:
        eta = ""
    print(f"{t:%H:%M} R4 {stage}: {done:,}/{TOTAL:,} pairs ({100 * done / TOTAL:.0f}%), {chunk}; "
          f"{rate:,} pairs/h over the last 30 min; {eta}")


if __name__ == "__main__":
    main()
