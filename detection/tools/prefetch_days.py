#!/usr/bin/env python3
"""Read-ahead for the R3 detection run: read the day files that the running detection processes will need next,
sequentially in large blocks, so they sit in the page cache when PhaseNet+ asks for them.

Why: the continuous archive is on a spinning RAID (sdb, mq-deadline scheduler, so I/O priorities have no effect).
The detection's data loaders read many day files at once in small requests (~70 KB): the disk spends its time seeking.
One reader taking whole files in 16 MB blocks, a few days ahead, turns that into sequential reads. Read only; it writes
nothing but its log line.

  python tools/prefetch_days.py [--ahead 3] [--interval 20]     (runs until no detection process is left)
"""
import argparse, glob, os, re, subprocess, sys, time

TREE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/continuous/sds_links")
PICKS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phasenet_plus_2020-2026/detection_location/{y}/picks")
BLOCK = 16 << 20


def running_years():
    out = subprocess.run(["ps", "-u", str(os.getuid()), "-o", "args="], capture_output=True, text=True).stdout
    return sorted({int(m.group(1)) for m in re.finditer(r"pipeline/detection\.py .*--year (\d{4})", out)})


def next_day(y):
    done = {int(f[-7:-4]) for f in glob.glob(os.path.join(PICKS.format(y=y), f"picks_{y}.*.csv"))}
    d = 1
    while d in done:
        d += 1
    return d


def read_file(p):
    try:
        fd = os.open(p, os.O_RDONLY)
    except OSError:
        return 0
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        n = 0
        while True:
            b = os.read(fd, BLOCK)
            if not b:
                return n
            n += len(b)
    finally:
        os.close(fd)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ahead", type=int, default=3); ap.add_argument("--interval", type=float, default=20)
    a = ap.parse_args()
    done = set(); t0 = time.time(); total = 0
    while True:
        ys = running_years()
        if not ys:
            print(f"[prefetch] no detection process left; read {total / 1e9:.1f} GB in {(time.time() - t0) / 3600:.1f} h", flush=True)
            return 0
        for y in ys:
            d0 = next_day(y)
            for d in range(d0, d0 + a.ahead + 1):
                if (y, d) in done:
                    continue
                files = sorted(glob.glob(os.path.join(TREE, f"*/*/*.D/*.{y}.{d:03d}")))
                ts = time.time(); n = sum(read_file(f) for f in files); total += n
                done.add((y, d))
                print(f"[prefetch] {time.strftime('%H:%M:%S')} {y}.{d:03d}: {len(files)} files, {n / 1e6:.0f} MB in {time.time() - ts:.0f} s", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
