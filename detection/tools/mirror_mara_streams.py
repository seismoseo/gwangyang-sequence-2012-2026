#!/usr/bin/env python
"""Renewal step R1b: fetch from MARA the continuous streams the 2026-10-11 station decisions add.

The local MARA mirror (data/continuous/KS_KG_mara_sds) holds one sensor per station, chosen by the old rule
HH > EL > HG (mirror_mara.py). The renewal's decisions need streams that rule never fetched; this tool fetches
exactly those, listed in STREAMS (docs/decisions/station_inventory_20261011 and ../ISSUES.md):
  - the accelerometer where the sensor rule chose it over the broadband: HANB, MOPB, BGDB, JODB, GOCB, BOSB, MUN
  - added stations: PMNM, YOA, HUK2 (both sensors, so the orientation can be measured and the rule applied), AMD
  - co-located second instruments: KWJU, NAHA, GOCBT; the 2023 temporary codes JAEBT, JEDAT, DAYAT
  - the KMA mobile stations of the 2020 response within 15 km: MHAW, PMAM, PMSM (added 2026-10-12)
Same SDS layout and the same rsync settings as mirror_mara.py (-aL dereferences MARA's symlinks; one job per
station-year-sensor; resumable). Files are only added; nothing already in the mirror changes, and the current
link layer (data/continuous/sds_links) does not point at the new streams.

  python detection/tools/mirror_mara_streams.py plan          (dry run: files and size per job, from rsync -n)
  python detection/tools/mirror_mara_streams.py pull --apply  (launch detached; idempotent)
  python detection/tools/mirror_mara_streams.py verify        (local files against MARA, per job)
  --only STA1,STA2 restricts any command to those stations
"""
import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # the project registry, found upward from this file
import argparse
import csv
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------- parameters (edit here)
MARA = "<user>@<mara-host>"
SDS = "<mara-sds-root>"
LOCAL = str(haenam.ROOT / "data/continuous/KS_KG_mara_sds")
OUT = os.path.join(LOCAL, "_manifest", "r1b_streams")
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4"]
# GWANGYANG PORT: one ssh connection per job, no ControlMaster (Imsil 2026-10-12: jobs multiplexed through one connection
# stalled at 0.2 MB/s for 7 h); keep-alives so a dead link fails instead of hanging
JOBS_DEFAULT = 4                 # GWANGYANG PORT: 4 jobs (was 2)
BWLIMIT_DEFAULT = 25000          # GWANGYANG PORT: kB/s per job (was 40000): 4 x 25 MB/s keeps moho's disk usable for others
STREAMS = []                     # GWANGYANG PORT: the Haenam R1b list is replaced by streams_r3x() below


# GWANGYANG PORT: every stream of the Gwangyang inventory (docs/station_inventory/tables): the chosen sensor of each station over
# its data years, plus the other sensors of stations where the sensor rule could not be applied yet (an orientation never
# measured: G2 measures them, then the rule decides). jobs() orders the station-years so that 2021 and 2026 come first.
def streams_r3x():
    import pandas as pd
    inv = os.path.join(str(haenam.ROOT), "docs/station_inventory/tables")
    st = pd.read_csv(os.path.join(inv, "stations.csv")); sp = pd.read_csv(os.path.join(inv, "stream_periods.csv"))
    rows = [(r.net, r.sta, r.chosen_sensor, int(r.first_day[:4]), int(r.last_day[:4])) for r in st.itertuples()]
    na = st[st.choice_status.str.startswith("current rule")]
    for r in na.itertuples():
        for q in sp[(sp.net == r.net) & (sp.sta == r.sta) & sp.sensor.isin(["HH", "EL", "HG"]) & (sp.sensor != r.chosen_sensor)].itertuples():
            rows.append((q.net, q.sta, q.sensor, int(q.first_day[:4]), int(q.last_day[:4])))
    return rows


def jobs(only=None, streams=None):
    keep = set(only.split(",")) if only else None
    js = [(net, sta, sen, y) for net, sta, sen, a, b in (streams or STREAMS) for y in range(a, b + 1) if keep is None or sta in keep]
    first = {2021: 0, 2026: 1}                                    # GWANGYANG PORT: the sequence years first, then 2012 .. 2025
    return sorted(js, key=lambda j: (first.get(j[3], 2), j[3], j[1], j[2]))


def rsync_cmd(net, sta, sen, y, bwlimit, dry):
    src = f"{MARA}:{SDS}/{y}/{net}/{sta}/"
    dst = f"{LOCAL}/{y}/{net}/{sta}/"
    cmd = ["rsync", "-aL", "--partial", f"--bwlimit={bwlimit}", "--info=stats2", "-e", "ssh " + " ".join(SSH_OPTS),
           f"--include={sen}?.D/", f"--include={sen}?.D/*", "--exclude=*", src, dst]
    return cmd[:1] + ["-n"] + cmd[1:] if dry else cmd


def stats(text):
    g = lambda k: int(re.search(rf"{k}: ([\d,]+)", text).group(1).replace(",", "")) if re.search(rf"{k}: ([\d,]+)", text) else 0
    return g("Number of regular files transferred"), g("Total transferred file size")


def run(job, bwlimit, dry):
    net, sta, sen, y = job
    os.makedirs(f"{LOCAL}/{y}/{net}/{sta}", exist_ok=True) if not dry else None
    t0 = time.time(); rc = None; out = ""
    for attempt in (1, 2):
        r = subprocess.run(rsync_cmd(net, sta, sen, y, bwlimit, dry), capture_output=True, text=True)
        rc, out = r.returncode, r.stdout + r.stderr
        if rc == 0 or rc not in (23, 24):
            break
    n, size = stats(out)
    if rc == 23 and "No such file" in out:
        rc = "absent"                                   # this station has no such sensor that year on MARA
    return dict(net=net, sta=sta, sensor=sen, year=y, rc=rc, files=n, bytes=size, seconds=round(time.time() - t0, 1),
                note="" if rc in (0, "absent") else out[-300:].replace("\n", " "))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("plan", "pull", "verify"):
        p = sub.add_parser(n); p.add_argument("--jobs", type=int, default=JOBS_DEFAULT)
        p.add_argument("--bwlimit", type=int, default=BWLIMIT_DEFAULT); p.add_argument("--apply", action="store_true")
        p.add_argument("--only", default="", help="comma-separated station codes (default: every stream)")
        p.add_argument("--set", default="r3x", choices=["", "r3x"], help="GWANGYANG PORT: r3x = the Gwangyang inventory streams (default)")
    a = ap.parse_args()
    dry = a.cmd in ("plan", "verify") or not a.apply
    js = jobs(a.only, streams_r3x() if a.set == "r3x" else None)
    print(f"{a.cmd}: {len(js)} station-sensor-years, {a.jobs} jobs, {'DRY RUN (rsync -n)' if dry else 'APPLY'}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    res = []
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(lambda j: run(j, a.bwlimit, dry), js):
            res.append(r)
            print(f"  {r['year']} {r['net']}.{r['sta']}.{r['sensor']}: rc {r['rc']}, files {r['files']:,}, "
                  f"{r['bytes']/1e9:.1f} GB {r['note'][:120]}", flush=True)
    name = {"plan": "plan.csv", "verify": "verify.csv", "pull": "pull_" + time.strftime("%Y%m%dT%H%M") + ".csv"}[a.cmd]
    if a.only:
        name = name.replace(".csv", "_" + a.only.replace(",", "-") + ".csv")
    if a.set:
        name = name.replace(".csv", f"_{a.set}.csv")
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    tot_f = sum(r["files"] for r in res); tot_b = sum(r["bytes"] for r in res)
    bad = [r for r in res if r["rc"] not in (0, "absent")]
    label = {"plan": "to fetch", "verify": "still missing", "pull": "fetched"}[a.cmd] if not (a.cmd == "pull" and dry) else "to fetch"
    print(f"TOTAL {label}: {tot_f:,} files, {tot_b/1e9:.1f} GB; problems {len(bad)}; report {os.path.join(OUT, name)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
