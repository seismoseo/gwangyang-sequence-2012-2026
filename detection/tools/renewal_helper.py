#!/usr/bin/env python3
"""Helper chain for the R3 run: once a chain has finished, its CPU cores pick up days from the END of the year that the
slowest remaining chain will reach last, one day at a time, working backwards. The main chain works forwards and skips
every day whose picks file already exists, so the two meet in the middle; at most the day they meet on is picked twice
(identical result). The main chain's locate step still runs after its own loop, so no day is ever missing from it.
Same pipeline, same settings, same command as the main chains (run_haenam.sh detect YEAR D-D), on the freed cores.

  python tools/renewal_helper.py --cores 48-63 [--after-year 2024]   (waits until that year's chain has finished)
"""
import argparse, glob, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PICKS = os.path.join(HERE, "phasenet_plus_2020-2026/detection_location/{y}/picks/picks_{y}.{d:03d}.csv")
LAST = {y: (273 if y == 2026 else (366 if y in (2020, 2024) else 365)) for y in range(2020, 2027)}
LOG = os.path.join(HERE, "logs/renewal_helper.log")
CLAIMS = os.path.join(HERE, "logs/helper_claims")          # one file per (year, day) a helper has taken: several helpers never share a day


def claim(y, d):
    os.makedirs(CLAIMS, exist_ok=True)
    try:
        os.close(os.open(os.path.join(CLAIMS, f"{y}.{d:03d}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)); return True
    except FileExistsError:
        return False


def say(msg):
    with open(LOG, "a") as f:
        f.write(time.strftime("%m-%d %H:%M:%S ") + msg + "\n")


def chains():
    return [[int(y) for y in c.split(",")] for c in open(os.path.join(HERE, "logs/renewal_chains.txt")).read().split()]


def undone(y):
    return [d for d in range(1, LAST[y] + 1) if not os.path.exists(PICKS.format(y=y, d=d))]


def running_years():
    out = subprocess.run(["ps", "-u", str(os.getuid()), "-o", "args="], capture_output=True, text=True).stdout
    return {int(m) for m in re.findall(r"pipeline/detection\.py .*--year (\d{4}) --days \d+-\d+", out) if "--days 1-" in out}


def next_day(tried):
    """the last undone day of the last unfinished year of the chain with the most undone days, behind its main frontier"""
    best = None
    for ch in chains():
        left = {y: undone(y) for y in ch}
        total = sum(len(v) for v in left.values())
        years = [y for y in ch if left[y]]
        if not years:
            continue
        y = years[-1]
        cand = [d for d in left[y] if (y, d) not in tried and not os.path.exists(os.path.join(CLAIMS, f"{y}.{d:03d}"))]
        if len(years) == 1:                           # the main chain is already in this year: stay 6 days ahead of its frontier
            frontier = left[y][0]
            cand = [d for d in cand if d > frontier + 6]          # > 15 min of main-chain progress away: no shared day
        if cand and (best is None or total > best[0]):
            best = (total, y, cand[-1])
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cores", required=True); ap.add_argument("--after-year", type=int, default=None)
    ap.add_argument("--name", default="helper")
    a = ap.parse_args()
    if a.after_year:
        say(f"waiting for the {a.after_year} chain to finish")
        while f"year {a.after_year} done" not in open(os.path.join(HERE, "logs/renewal_status.txt")).read():
            time.sleep(60)
    say(f"{a.name} starts on cores {a.cores}")
    tried = set()
    while True:
        if subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", "run_renewal_detection.sh years"], capture_output=True).returncode:
            say(f"driver gone: {a.name} stops"); return 0
        nd = next_day(tried)
        if nd is None:
            say(f"nothing left to take: {a.name} stops"); return 0
        _, y, d = nd
        tried.add((y, d))
        if not claim(y, d):                        # another helper took it a moment ago
            continue
        t0 = time.time()
        r = subprocess.run(["taskset", "-c", a.cores, "env", "OMP_WAIT_POLICY=PASSIVE", "WORKERS=4", "./run_haenam.sh", "detect", str(y), f"{d}-{d}", "run"],
                           cwd=HERE, capture_output=True, text=True)
        with open(os.path.join(HERE, f"logs/renewal_{y}_helper.log"), "a") as f:
            f.write(r.stdout + r.stderr)
        say(f"{a.name} {y}.{d:03d}: rc {r.returncode}, {'written' if os.path.exists(PICKS.format(y=y, d=d)) else 'no picks file'}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.exit(main())
