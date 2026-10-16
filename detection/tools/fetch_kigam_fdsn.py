#!/usr/bin/env python
"""Fill the KG (KIGAM) continuous-data gap after 2024-10 from the KIGAM Quake FDSN web service.

The MARA mirror (data/continuous/KS_KG_mara_sds) ends for every KG station on 2024-10-31. KIGAM Quake serves KG
station metadata (fdsnws/station) and waveforms (fdsnws/dataselect) from 2013 on, with no period or size limit
(https://data.kigam.re.kr/quake/data/api-service). Access needs the Geo Big Data Open Platform "open API" key, sent in
the request header X-Open-Api-Token. The key is read from TOKEN_FILE (or the environment variable KIGAM_OPENAPI_TOKEN)
and is never printed, logged or written anywhere else.

The key is registered for three addresses (<pago-host> = PAGO, <server-ip>, <server-ip>); moho (<moho-host>) is not
one of them, and KIGAM answers "Authentication failed (SSO result: 400)" to requests from moho. So the script opens an SSH
SOCKS tunnel to PAGO (TUNNEL_HOST) for the duration of a run and sends ONLY the KIGAM requests through it: KIGAM sees
PAGO's registered address, and every file is still written on moho. The tunnel is closed when the run ends.

Output: its own SDS tree, so the source of every file stays clear (the MARA mirror is not touched):
    data/continuous/KG_kigam_fdsn_sds/<YYYY>/KG/<STA>/<CHA>.D/KG.<STA>.<LOC>.<CHA>.D.<YYYY>.<DOY>
    data/continuous/KG_kigam_fdsn_sds/_manifest/   (plan, fetch log, verify and overlap reports, metadata comparison)
One file per channel and UTC day, trimmed to the day, miniSEED as delivered (re-written by ObsPy, encoding kept).
Only 100-sps streams are fetched (HH, EL, HG): BH/BG/SL are decimated copies of the same sensors.

Subcommands (DRY RUN by default; add --apply to write):
  metadata                 KG StationXML (level=response) for STATIONS -> data/stations/responses/master/KIGAM_KG_<date>.xml;
                           lists the channel epochs that start after the MARA archive ends (sensor changes since 2024-10)
  plan                     worklist (station, channel group, day) from START to END, skipping days already on disk
  overlap  [--days N]      fetch N days that the MARA mirror already holds (late October 2024) into a temporary folder
                           and compare them sample by sample with the MARA files: proves the two sources agree
  pull     [--jobs 2]      fetch the worklist; resumable (complete files are skipped), every day logged
  verify                   per station and channel: days expected, present, missing, with gaps
  status                   print the fetch log summary

Typical (run it yourself; one-time key file first):
  mkdir -p ~/.config/kigam && read -rs K && printf '%s' "$K" > ~/.config/kigam/openapi_token && chmod 600 ~/.config/kigam/openapi_token; unset K
  python detection/tools/fetch_kigam_fdsn.py metadata            (dry run: what would be fetched)
  python detection/tools/fetch_kigam_fdsn.py metadata --apply
  python detection/tools/fetch_kigam_fdsn.py overlap --apply      (2 days, compared with MARA)
  python detection/tools/fetch_kigam_fdsn.py plan --apply
  nohup python detection/tools/fetch_kigam_fdsn.py pull --apply > fetch_kigam.log 2>&1 &
  python detection/tools/fetch_kigam_fdsn.py verify --apply
"""
import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # the project registry, found upward from this file
import argparse
import csv
import datetime as dt
import hashlib
import os
import stat
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# ---------------------------------------------------------------- parameters (edit here)
SERVICE = "https://quake.kigam.re.kr/"
TOKEN_FILE = os.path.expanduser("~/.config/kigam/openapi_token")
TOKEN_ENV = "KIGAM_OPENAPI_TOKEN"
NETWORK = "KG"
# The six KG stations the Haenam products use (detection/station_table/stations.csv), plus YGN (82.5 km north; added by
# the user 2026-10-11 so that every station within 100 km is used; not on the MARA mirror, so its whole period comes
# from KIGAM). WDL (101 km, accelerometers only) is outside the 100-km rule.
STATIONS = ["JRB", "WDL", "GSU", "GRE", "KJM", "HWSB", "GHR", "IBA"]   # GWANGYANG PORT: the 8 KG stations of the Gwangyang station table
STATION_START = {"HWSB": "2026-10-01", "GHR": "2026-10-01", "IBA": "2026-10-01"}   # GWANGYANG PORT: MARA holds these three through
#   2026-09-30 (the Haenam renewal R1a KIGAM fill, delivered to MARA); the other five have MARA data only to 2024-10-31           # stations with no MARA data: fetch from the study start
CHANNEL_GROUPS = ["HH?", "EL?", "HG?"]          # 100-sps sensors; a group a station does not have is simply absent
def _chosen_groups():
    """GWANGYANG PORT: only the chosen sensor of each station (Sensor column of the station table), e.g. {"JRB": ["HG?"]}"""
    import pandas as _pd
    t = _pd.read_csv(haenam.path(None, "dl_station_table"))
    return {r.Code: [f"{r.Sensor}?"] for r in t[t.Network == NETWORK].itertuples()}
STATION_GROUPS = _chosen_groups()                # GWANGYANG PORT: per-station groups; CHANNEL_GROUPS only for stations not in the table
START = "2024-11-01"                            # the MARA mirror holds KG through 2024-10-31 (day 305)
END = "2026-10-06"                              # GWANGYANG PORT: the Gwangyang study end (last day of the 2026 MARA data)
OVERLAP_DAYS = ["2024-10-29", "2024-10-30"]     # days both sources hold, for the overlap check
EXPECTED_SPS = 100.0
JOBS_DEFAULT = 2                                # concurrent requests (be polite to the service)
RETRIES = 4
TIMEOUT_S = 300
TUNNEL_HOST = "msseo@<pago-host>"             # PAGO: a registered address that moho reaches by SSH key
TUNNEL_PORT = 18080                             # local SOCKS port of the tunnel
LOCAL = str(haenam.ROOT / "data/continuous/KG_kigam_fdsn_sds")
MARA_LOCAL = str(haenam.ROOT / "data/continuous/KS_KG_mara_sds")
MANIFEST_DIR = os.path.join(LOCAL, "_manifest")
MASTER_XML = str(haenam.ROOT / "data/stations/responses/master/KS_KG_metadata_1.0.2.xml")
META_OUT = str(haenam.ROOT / f"data/stations/responses/master/KIGAM_KG_{dt.date.today():%Y%m%d}.xml")
FETCH_LOG = os.path.join(MANIFEST_DIR, "fetch_log.csv")
WORKLIST = os.path.join(MANIFEST_DIR, "worklist.csv")
LOG_FIELDS = ["time", "station", "group", "day", "channel", "location", "status", "npts", "n_segments", "first", "last",
              "bytes", "sha256", "note"]


# ---------------------------------------------------------------- helpers
def token():
    t = os.environ.get(TOKEN_ENV, "").strip()
    if t:
        return t
    if not os.path.exists(TOKEN_FILE):
        sys.exit(f"no key: put the open-API key in {TOKEN_FILE} (chmod 600) or the environment variable {TOKEN_ENV}")
    mode = stat.S_IMODE(os.stat(TOKEN_FILE).st_mode)
    if mode & 0o077:
        sys.exit(f"{TOKEN_FILE} is readable by others (mode {oct(mode)}): run chmod 600 {TOKEN_FILE}")
    t = open(TOKEN_FILE).read().strip()
    if not t:
        sys.exit(f"{TOKEN_FILE} is empty")
    return t


class NoData(Exception):
    pass


class Kigam:
    """FDSN station/dataselect requests through an SSH SOCKS tunnel to TUNNEL_HOST (see the module docstring)."""

    def __init__(self):
        import subprocess, requests
        self.tok = token()
        self.proc = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
                                      "-N", "-D", f"127.0.0.1:{TUNNEL_PORT}", TUNNEL_HOST],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.px = {"https": f"socks5h://127.0.0.1:{TUNNEL_PORT}", "http": f"socks5h://127.0.0.1:{TUNNEL_PORT}"}
        self.s = requests.Session()
        for _ in range(30):                                       # wait for the tunnel (up to 15 s)
            time.sleep(0.5)
            if self.proc.poll() is not None:
                sys.exit(f"SSH tunnel to {TUNNEL_HOST} failed (exit {self.proc.returncode})")
            try:
                if self.s.get(SERVICE + "fdsnws/station/1/version", proxies=self.px, timeout=20).ok:
                    break
            except Exception:
                continue
        else:
            self.close(); sys.exit(f"SSH tunnel to {TUNNEL_HOST} did not come up")
        print(f"  tunnel up via {TUNNEL_HOST} (pid {self.proc.pid})", flush=True)

    def _get(self, path, params):
        r = self.s.get(SERVICE + path, params=params, headers={"X-Open-Api-Token": self.tok}, proxies=self.px, timeout=TIMEOUT_S)
        if r.status_code == 204 or (r.status_code == 404 and "no data" in r.text.lower()):
            raise NoData()
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        return r.content

    def get_stations(self, **kw):
        import io
        from obspy import read_inventory
        return read_inventory(io.BytesIO(self._get("fdsnws/station/1/query", dict(kw, format="xml"))))

    def get_waveforms(self, net, sta, loc, cha, t0, t1):
        import io
        from obspy import read
        b = self._get("fdsnws/dataselect/1/query", dict(network=net, station=sta, location=loc, channel=cha,
                                                         starttime=t0.strftime("%Y-%m-%dT%H:%M:%S"),
                                                         endtime=t1.strftime("%Y-%m-%dT%H:%M:%S")))
        if not b:
            raise NoData()
        return read(io.BytesIO(b), format="MSEED")

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()


def client():
    c = Kigam()
    import atexit
    atexit.register(c.close)
    return c


def days(a, b):
    d = dt.date.fromisoformat(a)
    e = dt.date.fromisoformat(b) if b else dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)
    while d <= e:
        yield d
        d += dt.timedelta(days=1)


def sds_path(root, sta, loc, cha, day):
    j = day.timetuple().tm_yday
    return os.path.join(root, f"{day.year}", NETWORK, sta, f"{cha}.D", f"{NETWORK}.{sta}.{loc}.{cha}.D.{day.year}.{j:03d}")


def local_days(root, sta, group, day):
    """files already on disk for this station, channel group (e.g. HH?) and day"""
    import glob
    j = day.timetuple().tm_yday
    return sorted(glob.glob(os.path.join(root, f"{day.year}", NETWORK, sta, f"{group[:2]}?.D",
                                         f"{NETWORK}.{sta}.*.{group[:2]}?.D.{day.year}.{j:03d}")))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log_rows(rows):
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    new = not os.path.exists(FETCH_LOG)
    with open(FETCH_LOG, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in LOG_FIELDS})


def fetch_day(c, sta, group, day, root, write=True):
    """one request per station, sensor group and UTC day; one file per channel. Returns log rows."""
    from obspy import UTCDateTime
    t0 = UTCDateTime(day.isoformat()); t1 = t0 + 86400
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    base = dict(time=now, station=sta, group=group, day=day.isoformat())
    st, err = None, ""
    for k in range(RETRIES):
        try:
            st = c.get_waveforms(NETWORK, sta, "*", group, t0, t1)
            break
        except NoData:
            return [dict(base, status="nodata")]
        except Exception as e:                                   # network or service error: back off and retry
            err = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(10 * (k + 1))
    if st is None:
        return [dict(base, status="error", note=err)]
    rows = []
    for (loc, cha) in sorted({(tr.stats.location, tr.stats.channel) for tr in st}):
        s = st.select(location=loc, channel=cha).copy()
        s.trim(t0, t1 - 1e-6, nearest_sample=False)
        s = s.__class__([tr for tr in s if tr.stats.npts > 0])
        if not len(s):
            rows.append(dict(base, channel=cha, location=loc, status="empty")); continue
        sps = {tr.stats.sampling_rate for tr in s}
        note = "" if sps == {EXPECTED_SPS} else f"sampling rate {sorted(sps)}"
        s.merge(method=-1)                                        # join contiguous records only; real gaps stay separate
        r = dict(base, channel=cha, location=loc, npts=sum(tr.stats.npts for tr in s), n_segments=len(s),
                 first=str(min(tr.stats.starttime for tr in s)), last=str(max(tr.stats.endtime for tr in s)), note=note)
        if write:
            p = sds_path(root, sta, loc, cha, day)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            tmp = p + ".part"
            s.write(tmp, format="MSEED")
            os.replace(tmp, p)
            r.update(status="ok", bytes=os.path.getsize(p), sha256=sha256(p))
        else:
            r.update(status="fetched")
        rows.append(r)
    return rows


# ---------------------------------------------------------------- subcommands
def cmd_metadata(a):
    print(f"metadata: KG StationXML (level=response) for {','.join(STATIONS)} from {SERVICE}")
    print(f"  output: {META_OUT}")
    if not a.apply:
        print("DRY RUN: nothing fetched or written (add --apply)"); return 0
    from obspy import read_inventory, UTCDateTime
    inv = client().get_stations(network=NETWORK, station=",".join(STATIONS), level="response")
    inv.write(META_OUT, format="STATIONXML")
    print(f"  wrote {META_OUT} ({os.path.getsize(META_OUT):,} bytes)")
    master = read_inventory(MASTER_XML).select(network=NETWORK)
    cut = UTCDateTime("2024-10-31")
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    out = os.path.join(MANIFEST_DIR, "metadata_compare.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["station", "channel", "start", "end", "depth_m", "sps", "sensitivity", "in_master"])
        for n in inv:
            for s in n:
                for ch in s:
                    m = master.select(station=s.code, channel=ch.code, location=ch.location_code)
                    known = any(abs(c.start_date - ch.start_date) < 86400 for nn in m for ss in nn for c in ss)
                    sens = ch.response.instrument_sensitivity.value if ch.response and ch.response.instrument_sensitivity else ""
                    w.writerow([s.code, ch.code, str(ch.start_date)[:10], str(ch.end_date)[:10] if ch.end_date else "",
                                ch.depth, ch.sample_rate, sens, known])
                    if ch.sample_rate == EXPECTED_SPS and (ch.start_date > cut or not known):
                        print(f"  NEW EPOCH {s.code}.{ch.code} from {str(ch.start_date)[:10]} depth {ch.depth} m "
                              f"({'not in the master XML' if not known else 'after the MARA archive end'})")
    print(f"  comparison: {out}")
    return 0


def latest_metadata():
    import glob
    fs = sorted(glob.glob(str(haenam.ROOT / "data/stations/responses/master/KIGAM_KG_*.xml")))
    return fs[-1] if fs else None


def worklist():
    """(station, sensor group, day) to request. With KIGAM metadata on disk (metadata --apply), only the groups whose
    vertical channel has an epoch covering the day; without it, every group every day (absent ones return 'nodata')."""
    meta = latest_metadata(); ep = None
    if meta:
        from obspy import read_inventory
        inv = read_inventory(meta)
        ep = {(s.code, ch.code[:2]): [] for n in inv for s in n for ch in s}
        for n in inv:
            for s in n:
                for ch in s:
                    if ch.code.endswith("Z") and ch.sample_rate == EXPECTED_SPS:
                        ep[(s.code, ch.code[:2])].append((ch.start_date.date, ch.end_date.date if ch.end_date else dt.date(9999, 1, 1)))
    rows = []
    for sta in STATIONS:
        for g in STATION_GROUPS.get(sta, CHANNEL_GROUPS):   # GWANGYANG PORT
            for d in days(STATION_START.get(sta, START), END):
                if ep is None or any(a <= d <= b for a, b in ep.get((sta, g[:2]), [])):
                    rows.append((sta, g, d))
    return rows, meta


def cmd_plan(a):
    wl, meta = worklist()
    todo = [(s, g, d) for (s, g, d) in wl if not local_days(LOCAL, s, g, d)]
    print(f"  channel epochs from: {meta or 'none (run metadata --apply first to skip sensors a station does not have)'}")
    end = END or str(dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1))
    print(f"plan: {len(STATIONS)} stations x {len(CHANNEL_GROUPS)} sensor groups, {START} .. {end} "
          f"(from {', '.join(f'{k} {v}' for k, v in STATION_START.items())}): "
          f"{len(wl):,} station-group-days, {len(todo):,} not yet on disk")
    print("  (a group a station does not have returns 'nodata' once per day and is logged as such; ~9 GB per station,"
          " sensor and year for 100-sps three-component data)")
    if a.apply:
        os.makedirs(MANIFEST_DIR, exist_ok=True)
        with open(WORKLIST, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["station", "group", "day"])
            for s, g, d in todo:
                w.writerow([s, g, d.isoformat()])
        print(f"  wrote {WORKLIST}")
    else:
        print("DRY RUN: worklist not written (add --apply)")
    return 0


def cmd_overlap(a):
    import glob
    from obspy import read
    print(f"overlap: fetch {OVERLAP_DAYS[:a.days]} from KIGAM and compare with the MARA mirror, sample by sample")
    if not a.apply:
        print("DRY RUN: nothing fetched (add --apply)"); return 0
    c = client(); res = []
    with tempfile.TemporaryDirectory() as tmp:
        for ds in OVERLAP_DAYS[:a.days]:
            d = dt.date.fromisoformat(ds)
            for sta in STATIONS:
                for g in STATION_GROUPS.get(sta, CHANNEL_GROUPS):   # GWANGYANG PORT
                    mara = local_days(MARA_LOCAL, sta, g, d)
                    if not mara:
                        continue
                    for r in fetch_day(c, sta, g, d, tmp, write=True):
                        if r.get("status") != "ok":
                            res.append((sta, r.get("channel", g), ds, r.get("status"), "")); continue
                        k = read(sds_path(tmp, sta, r["location"], r["channel"], d)).merge(method=-1)
                        m = [f for f in mara if f".{r['channel']}.D." in f]
                        if not m:
                            res.append((sta, r["channel"], ds, "not in MARA", "")); continue
                        mm = read(m[0]).merge(method=-1)
                        same = (len(k) == len(mm) and all(np.array_equal(x.data, y.data) and x.stats.starttime == y.stats.starttime
                                                          for x, y in zip(k, mm)))
                        res.append((sta, r["channel"], ds, "IDENTICAL" if same else "DIFFERENT",
                                    f"kigam {sum(t.stats.npts for t in k)} samples/{len(k)} seg, mara {sum(t.stats.npts for t in mm)}/{len(mm)}"))
    out = os.path.join(MANIFEST_DIR, "overlap_check.csv"); os.makedirs(MANIFEST_DIR, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["station", "channel", "day", "result", "detail"]); w.writerows(res)
    for r in res:
        print("  " + "  ".join(map(str, r)))
    n_id = sum(r[3] == "IDENTICAL" for r in res)
    print(f"RESULT: {n_id} of {len(res)} channel-days identical; report {out}")
    return 0 if res and n_id == len(res) else 1


def cmd_pull(a):
    if not os.path.exists(WORKLIST):
        sys.exit("no worklist: run 'plan --apply' first")
    wl = [(r["station"], r["group"], dt.date.fromisoformat(r["day"])) for r in csv.DictReader(open(WORKLIST))]
    todo = [(s, g, d) for (s, g, d) in wl if not local_days(LOCAL, s, g, d)]
    print(f"pull: {len(todo):,} station-group-days to fetch with {a.jobs} jobs")
    if not a.apply:
        print("DRY RUN: nothing fetched (add --apply)"); return 0
    c = client(); n_ok = n_fail = 0; t_start = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(fetch_day, c, s, g, d, LOCAL): (s, g, d) for (s, g, d) in todo}
        for i, f in enumerate(as_completed(futs), 1):
            rows = f.result(); log_rows(rows)
            n_ok += sum(r.get("status") == "ok" for r in rows); n_fail += sum(r.get("status") == "error" for r in rows)
            if i % 50 == 0 or i == len(todo):
                print(f"  {i:,}/{len(todo):,} requests; files written {n_ok:,}; errors {n_fail:,}; "
                      f"{(time.time() - t_start) / 3600:.2f} h", flush=True)
    print(f"done: files written {n_ok:,}; errors {n_fail:,} (re-run 'pull --apply' to retry the errors)")
    return 0 if n_fail == 0 else 1


def cmd_verify(a):
    import glob
    from obspy import read
    print(f"verify: {LOCAL}")
    rows = []
    for sta in STATIONS:
        for g in STATION_GROUPS.get(sta, CHANNEL_GROUPS):   # GWANGYANG PORT
            exp = list(days(STATION_START.get(sta, START), END)); have, gaps = 0, 0
            chans = set()
            for d in exp:
                fs = local_days(LOCAL, sta, g, d)
                if fs:
                    have += 1
                    for f in fs:
                        chans.add(os.path.basename(f).split(".")[3])          # NET.STA.LOC.CHA.D.YYYY.DDD
                        if a.apply:
                            gaps += max(0, len(read(f, headonly=True)) - 1)
            if have:
                rows.append((sta, g[:2], len(exp), have, len(exp) - have, gaps if a.apply else "", ",".join(sorted(chans))))
    for r in rows:
        print("  {}.{}: days {} present {} missing {} files-with-gaps {} channels {}".format(*r))
    if a.apply:
        out = os.path.join(MANIFEST_DIR, "verify.csv"); os.makedirs(MANIFEST_DIR, exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["station", "sensor", "days_expected", "days_present", "days_missing", "gap_segments", "channels"])
            w.writerows(rows)
        print(f"  report {out}")
    else:
        print("DRY RUN: day counts only; add --apply to read every file and count gaps")
    return 0


def cmd_status(a):
    if not os.path.exists(FETCH_LOG):
        print("no fetch log yet"); return 0
    from collections import Counter
    rows = list(csv.DictReader(open(FETCH_LOG)))
    cnt = Counter((r["station"], r["group"], r["status"]) for r in rows)
    for k in sorted(cnt):
        print(f"  {k[0]}.{k[1]} {k[2]}: {cnt[k]:,}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("metadata", "plan", "overlap", "pull", "verify", "status"):
        p = sub.add_parser(name)
        p.add_argument("--apply", action="store_true")
        if name == "pull":
            p.add_argument("--jobs", type=int, default=JOBS_DEFAULT)
        if name == "overlap":
            p.add_argument("--days", type=int, default=2)
    a = ap.parse_args()
    return {"metadata": cmd_metadata, "plan": cmd_plan, "overlap": cmd_overlap, "pull": cmd_pull,
            "verify": cmd_verify, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
