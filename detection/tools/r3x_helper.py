#!/usr/bin/env python3
"""Helper for the R3x run (2018-2019), as tools/renewal_helper.py is for R3: once one R3x chain has finished, its cores
pick days from the END of the day range with the most undone days, one day at a time, working backwards. The chain works
forwards and skips every day whose picks file exists, so the two meet; a helper stays more than 6 days ahead of the
chain's frontier, so no day is picked twice. Same command and settings as the chains (run_haenam.sh detect YEAR D-D run).

  python tools/r3x_helper.py --cores 0-15 --after "2018 --days 31-198"   (starts when that chain's detection has ended)
"""
import argparse, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PICKS = os.path.join(HERE, "phasenet_plus_2020-2026/detection_location/{y}/picks/picks_{y}.{d:03d}.csv")
RANGES = [(2018, 31, 198), (2018, 199, 365), (2019, 1, 182), (2019, 183, 365)]     # = run_r3x_detection.sh CHAINS_R3X
CLAIMS = os.path.join(HERE, "logs/r3x_helper_claims")
LOG = os.path.join(HERE, "logs/r3x_helper.log")


def say(msg):
    with open(LOG, "a") as f:
        f.write(time.strftime("%m-%d %H:%M:%S ") + msg + "\n")


def detecting(pattern):
    out = subprocess.run(["ps", "-u", str(os.getuid()), "-o", "args="], capture_output=True, text=True).stdout
    return any("pipeline/detection.py" in l and pattern in l for l in out.splitlines())


def claim(y, d):
    os.makedirs(CLAIMS, exist_ok=True)
    try:
        os.close(os.open(os.path.join(CLAIMS, f"{y}.{d:03d}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)); return True
    except FileExistsError:
        return False


def next_day():
    best = None
    for y, d0, d1 in RANGES:
        undone = [d for d in range(d0, d1 + 1) if not os.path.exists(PICKS.format(y=y, d=d))]
        if not undone:
            continue
        frontier = undone[0]                      # the chain works forwards from here
        cand = [d for d in undone if d > frontier + 6 and not os.path.exists(os.path.join(CLAIMS, f"{y}.{d:03d}"))]
        if cand and (best is None or len(undone) > best[0]):
            best = (len(undone), y, cand[-1])
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cores", required=True); ap.add_argument("--after", required=True)
    a = ap.parse_args()
    say(f"waiting for the chain '{a.after}' to finish (cores {a.cores})")
    while detecting(f"--year {a.after}"):
        time.sleep(60)
    say(f"helper on cores {a.cores} starts")
    while True:
        nd = next_day()
        if nd is None:
            say(f"nothing left to take: helper on {a.cores} stops"); return 0
        _, y, d = nd
        if not claim(y, d):
            continue
        t0 = time.time()
        r = subprocess.run(["taskset", "-c", a.cores, "env", "OMP_WAIT_POLICY=PASSIVE", "WORKERS=4", "./run_haenam.sh", "detect", str(y), f"{d}-{d}", "run"],
                           cwd=HERE, capture_output=True, text=True)
        with open(os.path.join(HERE, f"logs/r3x_{y}_helper.log"), "a") as f:
            f.write(r.stdout + r.stderr)
        say(f"{y}.{d:03d}: rc {r.returncode}, {'written' if os.path.exists(PICKS.format(y=y, d=d)) else 'no picks file'}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.exit(main())
