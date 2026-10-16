#!/usr/bin/env python3
"""Progress of the R3 detection run: days picked per year, the rate over the last hour, the expected end of each chain,
the locate stage, errors in the logs, and the state of the processes and the disk. One line (default) or a table (--table).

  python tools/renewal_progress.py [--table]
"""
import argparse, glob, os, re, subprocess, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PICKS = os.path.join(HERE, "phasenet_plus_2020-2026/detection_location/{y}/picks")
LOGS = os.path.join(HERE, "logs")
TARGET = {y: (273 if y == 2026 else (366 if y in (2020, 2024) else 365)) for y in range(2020, 2027)}
_cf = os.path.join(LOGS, "renewal_chains.txt")
CHAINS = ([[int(y) for y in c.split(",")] for c in open(_cf).read().split()] if os.path.exists(_cf)
          else [[2020, 2022, 2024, 2026], [2021, 2023, 2025]])
ERR = re.compile(r"Traceback|Error|Killed|CUDA out of memory|Segmentation|ABORT", re.I)


def year_state(y):
    fs = glob.glob(os.path.join(PICKS.format(y=y), f"picks_{y}.*.csv"))
    now = time.time()
    recent = sum(1 for f in fs if now - os.path.getmtime(f) < 3600)
    recent2 = sum(1 for f in fs if now - os.path.getmtime(f) < 7200)
    days = {int(f[-7:-4]) for f in fs}
    n = len([d for d in days if d <= TARGET[y]])
    loc = os.path.exists(os.path.join(LOGS, f"renewal_{y}_locate.log"))
    loc_done = loc and "HYPOINVERSE" in open(os.path.join(LOGS, f"renewal_{y}_locate.log"), errors="ignore").read() and n >= TARGET[y]
    errs = []
    for f in glob.glob(os.path.join(LOGS, f"renewal_{y}_*.log")):
        errs += [l.strip()[:120] for l in open(f, errors="ignore") if ERR.search(l)]
    return dict(year=y, done=n, target=TARGET[y], last_hour=recent, last_2h=recent2, locating=loc and not loc_done, errors=errs)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--table", action="store_true"); a = ap.parse_args()
    st = {y: year_state(y) for y in range(2020, 2027)}
    procs = subprocess.run(["ps", "-u", str(os.getuid()), "-o", "args="], capture_output=True, text=True).stdout
    running = sorted({int(m) for m in re.findall(r"pipeline/detection\.py .*--year (\d{4})", procs)})
    locating = sorted({int(m) for m in re.findall(r"run_pipeline\.py .*--years (\d{4})", procs)})
    driver = "run_renewal_detection.sh years" in procs
    prefetch = "prefetch_days.py" in procs
    etas = []
    for ch in CHAINS:
        left = sum(st[y]["target"] - st[y]["done"] for y in ch)
        rate = sum(st[y]["last_2h"] for y in ch) / 2.0                 # days per hour of the chain over the last 2 h (across a year change)
        etas.append(0.0 if left <= 0 else (left / rate if rate else float("inf")))
    tot_done = sum(s["done"] for s in st.values()); tot = sum(s["target"] for s in st.values())
    try:
        io = subprocess.run(["iostat", "-x", "sdb", "3", "2"], capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[-1].split()
        disk = f"disk {float(io[-1]):.0f}% busy, {float(io[3]) / 1e3:.0f} MB/s read"
    except Exception:
        disk = "disk ?"
    errs = [(y, e) for y in st for e in st[y]["errors"]]
    line = (f"{time.strftime('%m-%d %H:%M')} R3: {tot_done}/{tot} days ({100 * tot_done / tot:.1f}%); "
            + "; ".join(f"{y} {st[y]['done']}/{st[y]['target']}" for y in range(2020, 2027) if st[y]["done"] or y in running)
            + f" | detecting {running or '-'}, locating {locating or '-'} | rate {sum(max(st[y]['last_hour'] for y in ch) for ch in CHAINS)} days/h"
            + " | chains end " + ", ".join(time.strftime('%a %H:%M', time.localtime(time.time() + 3600 * e)) if e < 1e6 else '?' for e in etas)
            + f" | per-chain end without helpers ~{time.strftime('%a %H:%M', time.localtime(time.time() + 3600 * max(etas))) if max(etas) < 1e6 else '?'}"
            + (lambda r: f" | ALL DETECTION DONE ~{time.strftime('%a %m-%d %H:%M', time.localtime(time.time() + 3600 * (tot - tot_done) / r)) if r else '?'} (all cores, {r:.0f} days/h over 2 h)")(
                sum(st[y]["last_2h"] for y in st) / 2.0)
            + f" | {disk} | prefetch {'on' if prefetch else 'off'}"
            + ("" if driver else " | DRIVER NOT RUNNING") + (f" | ERRORS {len(errs)}: {errs[-1][0]} {errs[-1][1]}" if errs else ""))
    fl = os.path.join(os.path.dirname(HERE), "data/continuous/KS_KG_mara_sds/_manifest/r1b_streams/pull_r3x.log")
    if os.path.exists(fl):
        t = open(fl, errors="ignore").read()
        jobs = re.findall(r"rc (\S+), files [\d,]+, ([\d.]+) GB", t)
        bad = [j for j in jobs if j[0] not in ("0", "absent")]
        line += f" | 2018-19 fetch {len(jobs)}/164 jobs, {sum(float(g) for _, g in jobs):.0f} GB" + (f", {len(bad)} FAILED" if bad else "") + (" DONE" if "TOTAL" in t else "")
    print(line)
    if a.table:
        for y in range(2020, 2027):
            s = st[y]; print(f"  {y}: {s['done']:3d}/{s['target']} days, {s['last_hour']} in the last hour{', locating' if s['locating'] else ''}"
                              + (f", {len(s['errors'])} error lines" if s["errors"] else ""))


if __name__ == "__main__":
    main()
