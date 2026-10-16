#!/usr/bin/env python
"""GWANGYANG (decision G-D13): re-fetch from NECIS the KS components missing on MARA inside the 2021 episode, guarded.

Replaces necis_refetch_episodes.sh (2026-10-15 17:44), whose job 4 showed two failure modes of the NECIS downloader
(/home/msseo/works/Claude/archive_necis.sh -> download_continuous.py):
  * "Station checkboxes not found in DOM - selecting all by default": the request goes out for EVERY station (2.6 GB for
    one Z channel-day), and the organizer would unpack every station's file into necis_refetch, the link tool's FIRST source,
    silently replacing good MARA files of that day;
  * "Modal confirm failed" / "request failed": nothing is requested.
So each job runs in its own process group with its own log; on either message the job is killed, its partial zip removed,
and the job retried (up to MAX_TRY times). After a clean finish, only files of the requested stations and channels are kept;
any other new file is moved to data/continuous/necis_refetch/rejected/ (never linked). A clean finish with 0 files means NECIS
has no data for that request.
Not requested (see the .sh header): BOSB HHZ, GMNA HGZ on 2021-08-27; MALA 2021-07-10 (asked 17:44, NECIS has none); KG WDL;
HWJA 2026-01-03..05-09 (outage). JINA HGZ 2021-08-08 was recovered by the .sh run (job 3).
    python detection/tools/necis_refetch_episodes.py            (lists the jobs)
    python detection/tools/necis_refetch_episodes.py run        (runs them; summary in logs/necis_refetch_jobs_20261015.csv)
"""
import os, re, signal, subprocess, sys, time
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data/continuous/necis_refetch/continuous")
REJECT = os.path.join(ROOT, "data/continuous/necis_refetch/rejected")
REQ = os.path.join(ROOT, "data/continuous/necis_refetch/requests")
LOGDIR = os.path.join(ROOT, "logs/necis_refetch_attempts")
SUMMARY = os.path.join(ROOT, "logs/necis_refetch_jobs_20261015.csv")
ARCHIVE = "/home/msseo/works/Claude/archive_necis.sh"
ZIPS = "/home/msseo/works/Claude/data/necis/zips"
BAD = ("selecting all by default", "Modal confirm failed", "request failed")
MAX_TRY = 3
JOBS = [("2021-07-02", "KUJA", "HGZ,HGN,HGE", 2),
        ("2021-08-20", "DUSB", "HHZ", 5),
        ("2021-08-20", "GWYB", "HHZ,HHN,HHE", 2),
        ("2021-08-24", "HAMB,DUSB", "HHZ,HHN,HHE", 2),
        ("2021-08-24", "WICA,JNYA,GDDB", "HGZ,HGN,HGE", 2),
        ("2021-08-26", "JAGA,NAJA", "HGZ", 5),
        ("2021-08-26", "JEU2", "ELZ", 5),
        ("2021-08-27", "BAU", "ELZ,ELN,ELE", 2),
        ("2021-08-28", "GWYB", "HHZ", 5),
        ("2021-08-28", "BAU", "ELZ", 5),
        ("2021-08-28", "BLLA,GMNA,NAHA,SCHA,YESA", "HGZ", 5)]


def files_now():
    out = set()
    for d, _, fs in os.walk(OUT):
        out |= {os.path.join(d, f) for f in fs}
    return out


def wanted(path, day, stas, chans):
    m = re.match(r"KS\.([A-Z0-9]+)\.([A-Z0-9]{3})\.(\d{4})\.(\d{3})\.", os.path.basename(path))
    if not m:
        return False
    doy = pd.Timestamp(day).dayofyear
    return m.group(1) in stas and m.group(2) in chans and int(m.group(3)) == pd.Timestamp(day).year and int(m.group(4)) == doy


def attempt(n, k, day, stas, chans, batch):
    csv = os.path.join(REQ, f"stations_v2_{day.replace('-', '')}_{n}.csv")
    st = pd.read_csv(os.path.join(ROOT, "detection/station_table/stations.csv")).set_index("Code")
    pd.DataFrame([dict(network="KS", station=s, startdate="1/1/12", latitude=st.Latitude[s], longitude=st.Longitude[s],
                       datetime_start="2012-01-01") for s in stas]).to_csv(csv, index=False)
    log = os.path.join(LOGDIR, f"job{n:02d}_try{k}.log")
    env = dict(os.environ, CONTINUOUS_DIR=OUT, STATION_CSV=csv, CHANNELS=",".join(chans), BATCH_SIZE=str(batch))
    with open(log, "w") as fh:
        p = subprocess.Popen([ARCHIVE, day, day], stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    bad = None
    while p.poll() is None:
        time.sleep(5)
        txt = open(log, errors="replace").read()
        bad = next((b for b in BAD if b in txt), None)
        if bad:
            os.killpg(p.pid, signal.SIGTERM); time.sleep(5)
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
            for z in re.findall(r"Downloading \S+/(\S+\.zip)", txt):          # this attempt's partial zip only
                for f in (os.path.join(ZIPS, z), os.path.join(ZIPS, z + ".part")):
                    if os.path.exists(f):
                        os.remove(f)
            break
    if bad is None:
        txt = open(log, errors="replace").read()
        bad = next((b for b in BAD if b in txt), None)
    return bad, p.returncode, log


def main():
    run = len(sys.argv) > 1 and sys.argv[1] == "run"
    for n, (day, stas, chans, batch) in enumerate(JOBS, 1):
        print(f"job {n}: {day} {stas} {chans} (batch {batch})")
    if not run:
        print("dry run: add 'run' to execute"); return
    os.makedirs(LOGDIR, exist_ok=True); os.makedirs(REJECT, exist_ok=True)
    rows = []
    for n, (day, stas, chans, batch) in enumerate(JOBS, 1):
        stas, chans = stas.split(","), chans.split(",")
        for k in range(1, MAX_TRY + 1):
            before = files_now()
            bad, rc, log = attempt(n, k, day, stas, chans, batch)
            new = sorted(files_now() - before)
            keep = [f for f in new if wanted(f, day, stas, chans)]
            rej = [f for f in new if f not in keep]
            for f in rej:
                os.replace(f, os.path.join(REJECT, os.path.basename(f)))
            print(f"{time.strftime('%F %T')} job {n} try {k}: {'ABORTED (' + bad + ')' if bad else 'finished'} rc {rc}; "
                  f"kept {len(keep)}, rejected {len(rej)}", flush=True)
            if not bad:
                break
        rows.append(dict(job=n, day=day, stations=",".join(stas), channels=",".join(chans), tries=k,
                         outcome="aborted" if bad else ("recovered" if keep else "none at NECIS"),
                         kept=" ".join(os.path.basename(f) for f in keep), n_rejected=len(rej), last_log=os.path.relpath(log, ROOT)))
        pd.DataFrame(rows).to_csv(SUMMARY, index=False)
    print(pd.DataFrame(rows)[["job", "day", "stations", "channels", "tries", "outcome", "n_rejected"]].to_string(index=False))
    print("ALL_JOBS_DONE", time.strftime("%F %T"), flush=True)


if __name__ == "__main__":
    main()
