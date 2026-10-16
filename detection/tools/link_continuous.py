#!/usr/bin/env python3
"""Build / refresh / check the continuous-data symlink tree in the Buan-pipeline layout.

WHY: the validated Buan pipeline (20.2024_Buan_EQ_DL_detection) and the template-matching code both discover day files as
    <tree>/<NET>/<STA>/<CHA>.D/<NET>.<STA>..<CHA>.D.<YYYY>.<DOY>      (7 dot-fields, asserted)
whereas the Haenam data live in different layouts:
    repaired day files : data/continuous/repaired_20261013/<YYYY>/<STA>/<NET>.<STA>.<CHA>.<YYYY>.<DOY>.00.00.00 (damaged records dropped)
    NECIS re-fetch     : data/continuous/necis_refetch_20261013/continuous/<YYYY>/<STA>/<NET>.<STA>.<CHA>.<YYYY>.<DOY>.HH.MM.SS
    MARA SDS mirror    : data/continuous/KS_KG_mara_sds/<YYYY>/<NET>/<STA>/<CHA>.D/<NET>.<STA>..<CHA>.D.<YYYY>.<DOY>
    NECIS cron archive : /home/msseo/works/Claude/data/necis/continuous/<YYYY>/<STA>/KS.<STA>.<CHA>.<YYYY>.<DOY>.HH.MM.SS
    JNU tar extracts   : data/jnu/mseed/JU/<STA>/<STA>.JU.00.<CHA>.<YYYY>.<DOY>
    Earthworm archive  : ~/EW/EW_JNU/data/mseed/JU/<STA>/<STA>.JU.00.<CHA>.<YYYY>.<DOY>
Rather than patching the validated code, every source is exposed through ONE symlink tree (registry product
dl_continuous: data/continuous/sds_links). Nothing is copied; nothing outside that tree is written.
Re-run whenever the archives grow (the NECIS cron adds a day every night; Earthworm every day).

Only stations listed in the station table (registry product dl_station_table) are linked (the 100-km set), and of each
station only its chosen sensor (R2), the Sensor column of the table: one sensor per station for the whole window,
chosen by the user's decisions and sensor rule (docs/decisions/station_inventory_20261011). Every pipeline that reads
this tree therefore uses exactly that sensor. Where the same (station, channel, day) exists in several sources the
FIRST source in SOURCES wins (JNU tar extracts are complete days; Earthworm days may be partial while live).

Paths come from the registry (registry/haenam.py): the project root is haenam.ROOT, the tree and the station table are
the products dl_continuous and dl_station_table; the in-project sources are composed from haenam.ROOT. The NECIS and
Earthworm archives are outside the project and are only read.

Usage (from the project root; dry run by default):
    python detection/tools/link_continuous.py                          # DRY RUN: counts what would be linked
    python detection/tools/link_continuous.py --apply                  # create the links
    python detection/tools/link_continuous.py --apply --years 2026     # restrict to years
    python detection/tools/link_continuous.py --apply --refresh        # re-point links whose target or form changed
    python detection/tools/link_continuous.py --relative ...           # links to in-project sources are written
                                                                       # relative (../../..); external ones stay absolute
    python detection/tools/link_continuous.py --check [--relative] [--min-links N]
        read-only: every link resolves (zero dangling), every planned day file has its link and every link is planned,
        each link points at the planned file, in-project links are relative when --relative is given, and there are at
        least --min-links links (default MIN_LINKS, the count of PLAN P7 step 3). Exit status 1 on any failure.
    --target DIR checks or builds another tree (e.g. a scratch copy of the link layer) instead of dl_continuous.
P7 step 3: python detection/tools/link_continuous.py --relative --apply, then --relative --check.
To add another archive, append a SOURCES entry with its filename regex and re-run with --apply.
"""
import argparse
import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "registry"))
import haenam as H  # noqa: E402

# --------------------------------------------------------------------------- parameters
PROJECT = str(H.ROOT)
MIN_LINKS = 0                          # renewal: the census is the plan itself (--check compares link by link)
NECIS_ROOT = os.environ.get("HAENAM_NECIS_CONTINUOUS", "/home/msseo/works/Claude/data/necis/continuous")   # read only
EW_ROOT = os.environ.get("HAENAM_EW_MSEED", os.path.expanduser("~/EW/EW_JNU/data/mseed"))                  # read only


def product(name, default_rel):
    """(path, how): the registry product, or PROJECT/default_rel when the registry has no such product (printed)."""
    try:
        return str(H.path(None, name)), f"registry product {name}"
    except KeyError:
        return os.path.join(PROJECT, default_rel), f"default {default_rel} (registry has no product {name})"


TARGET, TARGET_HOW = product("dl_continuous", "data/continuous/sds_links")
STATIONS_CSV, STATIONS_HOW = product("dl_station_table", "detection/station_table/stations.csv")

# --------------------------------------------------------------------------- sources
# Each source: root glob for day files + a regex with named groups sta, cha, year, doy (+ optional net).
# `net` fixed when the filename does not carry it reliably.
SOURCES = [
    dict(name="repaired",                  # day files with damaged records dropped (2021-03-03; tools/repair_mseed_records.py,
         glob=os.path.join(PROJECT, "data/continuous/repaired_20261013/*/*/*"),   # user decision 2026-10-13): first
         regex=r"(?P<net>[A-Z]{2})\.(?P<sta>[A-Z0-9]+)\.(?P<cha>[A-Z0-9]{3})\.(?P<year>\d{4})\.(?P<doy>\d{3})\.\d\d\.\d\d\.\d\d$",
         net=None),
    dict(name="necis_refetch",             # clean NECIS copies of day files corrupted in the MARA archive (2026-10-13): first
         glob=os.path.join(PROJECT, "data/continuous/necis_refetch/continuous/*/*/*"),   # GWANGYANG PORT: G-D13 re-fetches
         regex=r"(?P<net>[A-Z]{2})\.(?P<sta>[A-Z0-9]+)\.(?P<cha>[A-Z0-9]{3})\.(?P<year>\d{4})\.(?P<doy>\d{3})\.\d\d\.\d\d\.\d\d$",
         net=None),
    dict(name="mara",                      # MARA SDS mirror: KS 2020-2026, KG 2020-2024; first source wins
         glob=os.path.join(PROJECT, "data/continuous/KS_KG_mara_sds/*/*/*/*.D/*"),
         regex=r"^(?P<net>[A-Z]{2})\.(?P<sta>[A-Z0-9]+)\.\.(?P<cha>[A-Z0-9]{3})\.D\.(?P<year>\d{4})\.(?P<doy>\d{3})$",
         net=None),
    dict(name="kigam",                     # KIGAM FDSN fill of KG 2024-11-01 .. 2026-09-30 (renewal step R1a)
         glob=os.path.join(PROJECT, "data/continuous/KG_kigam_fdsn_sds/*/*/*/*.D/*"),
         regex=r"^(?P<net>[A-Z]{2})\.(?P<sta>[A-Z0-9]+)\.\.(?P<cha>[A-Z0-9]{3})\.D\.(?P<year>\d{4})\.(?P<doy>\d{3})$",
         net=None),
    dict(name="necis",
         glob=os.path.join(NECIS_ROOT, "*/*/*"),
         regex=r"(?P<net>[A-Z]{2})\.(?P<sta>[A-Z0-9]+)\.(?P<cha>[A-Z0-9]{3})\.(?P<year>\d{4})\.(?P<doy>\d{3})\.\d\d\.\d\d\.\d\d$",
         net=None),
    dict(name="jnu_tar",
         glob=os.path.join(PROJECT, "data/jnu/mseed/JU/*/*"),
         regex=r"(?P<sta>[A-Z0-9]+)\.(?P<net>JU)\.00\.(?P<cha>[A-Z0-9]{3})\.(?P<year>\d{4})\.(?P<doy>\d{3})$",
         net="JU"),
    dict(name="earthworm",
         glob=os.path.join(EW_ROOT, "JU/*/*"),
         regex=r"(?P<sta>[A-Z0-9]+)\.(?P<net>JU)\.00\.(?P<cha>[A-Z0-9]{3})\.(?P<year>\d{4})\.(?P<doy>\d{3})$",
         net="JU"),
]


def wanted_stations():
    """(net, sta) -> the chosen sensor (R2): the Sensor column of the station table"""
    st = pd.read_csv(STATIONS_CSV)
    return {(r.Network, r.Code): r.Sensor for r in st.itertuples()}


def link_until():
    """(net, sta) -> last day to link (YYYY, DOY), from the optional LinkUntil column: a predecessor code ends the day
    before its successor at the same site starts, so no site is counted twice (R3x, 2026-10-13)"""
    st = pd.read_csv(STATIONS_CSV)
    if "LinkUntil" not in st:
        return {}
    out = {}
    for r in st.itertuples():
        if isinstance(r.LinkUntil, str) and r.LinkUntil:
            t = pd.Timestamp(r.LinkUntil); out[(r.Network, r.Code)] = (t.year, t.dayofyear)
    return out


def scan(years=None):
    """Yield (net, sta, cha, year, doy, path, source) for every day file in every source."""
    for src in SOURCES:
        rx = re.compile(src["regex"])
        for p in glob.iglob(src["glob"]):
            m = rx.search(os.path.basename(p))
            if not m:
                continue
            net = src["net"] or m.group("net")
            year = int(m.group("year"))
            if years and year not in years:
                continue
            yield net, m.group("sta"), m.group("cha"), year, int(m.group("doy")), p, src["name"]


def under(p, r):
    return p == r or p.startswith(r.rstrip("/") + "/")


def link_text(path, link_dir, relative):
    """The text of the link at link_dir to path: relative for an in-project path under --relative, else absolute."""
    if relative and under(os.path.abspath(path), PROJECT):
        return os.path.relpath(os.path.abspath(path), link_dir)
    return path


def plan(years, want):
    planned, seen, n_skip = {}, {}, 0
    until = link_until()
    for net, sta, cha, year, doy, path, srcname in scan(years):
        if (net, sta) not in want or cha[:2] != want[(net, sta)]:          # not in the table, or not its chosen sensor (R2)
            n_skip += 1
            continue
        if (net, sta) in until and (year, doy) > until[(net, sta)]:         # after a predecessor code's last day (R3x)
            n_skip += 1
            continue
        key = (net, sta, cha, year, doy)
        if key in planned:                       # first source wins
            continue
        planned[key] = (path, srcname)
        seen[srcname] = seen.get(srcname, 0) + 1
    # GWANGYANG PORT (decision G-D7): a temporary code merged into its site (registry/streams.csv role "temporary code of SITE")
    # is linked only on days when the site's own chosen sensor has no file, so no site is counted twice
    merged = {}
    sp = os.path.join(PROJECT, "registry/streams.csv")
    if os.path.exists(sp):
        for r in pd.read_csv(sp).itertuples():
            m = re.match(r"temporary code of ([A-Z0-9]+)", str(r.role))
            if m:
                merged[(r.net, r.sta)] = (r.net, m.group(1))
    site_days = {(k[0], k[1], k[3], k[4]) for k in planned}
    for k in [k for k in planned if (k[0], k[1]) in merged]:
        snet, ssta = merged[(k[0], k[1])]
        if (snet, ssta, k[3], k[4]) in site_days:
            src = planned.pop(k)[1]; seen[src] -= 1; n_skip += 1
    return planned, seen, n_skip


def link_path(target, net, sta, cha, year, doy):
    return os.path.join(target, net, sta, f"{cha}.D", f"{net}.{sta}..{cha}.D.{year}.{doy:03d}")


def check(target, planned, relative, min_links, years):
    """Read-only census of the tree against the plan. Returns the number of failures."""
    want = {link_path(target, *k): v[0] for k, v in planned.items()}
    n = dangling = wrong_target = absolute_inproject = not_link = extra = 0
    examples = {}
    for dp, dns, fns in os.walk(target):
        for f in fns + [d for d in dns if os.path.islink(os.path.join(dp, d))]:
            p = os.path.join(dp, f)
            if not os.path.islink(p):
                not_link += 1
                examples.setdefault("not a link", p)
                continue
            m = re.search(r"\.(\d{4})\.\d{3}$", f)
            if years and m and int(m.group(1)) not in years:
                continue
            n += 1
            t = os.readlink(p)
            if not os.path.exists(p):
                dangling += 1
                examples.setdefault("dangling", f"{p} -> {t}")
            if relative and os.path.isabs(t) and under(t, PROJECT):
                absolute_inproject += 1
                examples.setdefault("absolute link into the project", f"{p} -> {t}")
            src = want.get(p)
            if src is None:
                extra += 1
                examples.setdefault("not in the plan", p)
            elif os.path.realpath(p) != os.path.realpath(src):
                wrong_target += 1
                examples.setdefault("points elsewhere than planned", f"{p} -> {t} (planned {src})")
    missing = sum(1 for p in want if not os.path.lexists(p))
    if missing:
        examples.setdefault("planned but absent", next(p for p in want if not os.path.lexists(p)))
    fails = {"dangling": dangling, "points elsewhere than planned": wrong_target, "not in the plan": extra,
             "planned but absent": missing, "not a link": not_link,
             "absolute link into the project": absolute_inproject if relative else 0,
             f"fewer than {min_links} links": int(n < min_links)}
    print(f"[CHECK] {target}: {n} links; planned {len(want)}")
    for k, v in fails.items():
        print(f"  {k:32s} {v}" + (f"   e.g. {examples[k]}" if v and k in examples else ""))
    nf = sum(1 for v in fails.values() if v)
    print(f"  RESULT: {'PASS' if nf == 0 else 'FAIL'}")
    return nf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="create the symlinks (default: dry run)")
    ap.add_argument("--refresh", action="store_true", help="re-point links whose target or form differs from the plan")
    ap.add_argument("--years", type=str, default=None, help="comma list, e.g. 2026 or 2020,2026")
    ap.add_argument("--relative", action="store_true", help="write in-project links relative (P7)")
    ap.add_argument("--check", action="store_true", help="read-only census of the tree (exit 1 on any failure)")
    ap.add_argument("--min-links", type=int, default=MIN_LINKS)
    ap.add_argument("--target", default=None, help="the link tree (default: registry product dl_continuous)")
    a = ap.parse_args()
    years = {int(y) for y in a.years.split(",")} if a.years else None
    target = os.path.abspath(a.target) if a.target else TARGET
    print(f"project root {PROJECT}; tree {target} ({'--target' if a.target else TARGET_HOW}); "
          f"stations {STATIONS_CSV} ({STATIONS_HOW})")
    target_is_link = os.path.lexists(target) and os.path.realpath(target) != os.path.normpath(target)
    parent_is_link = os.path.realpath(os.path.dirname(target)) != os.path.dirname(target)
    if target_is_link or parent_is_link:       # e.g. a rehearsal clone whose tree is a link to the live one
        sys.exit(f"REFUSED: {target} has a symlink on its path (it would write elsewhere: "
                 f"{os.path.realpath(target)})")
    want = wanted_stations()
    planned, seen, n_skip_station = plan(years, want)
    if a.check:
        sys.exit(1 if check(target, planned, a.relative, a.min_links, years) else 0)

    n_new = n_exist = n_repoint = 0
    for (net, sta, cha, year, doy), (path, srcname) in sorted(planned.items()):
        link = link_path(target, net, sta, cha, year, doy)
        d = os.path.dirname(link)
        text = link_text(path, d, a.relative)
        if os.path.lexists(link):
            same_file = os.path.realpath(link) == os.path.realpath(path)
            if same_file and os.readlink(link) == text:
                n_exist += 1
                continue
            if not a.refresh:
                n_exist += 1
                continue
            n_repoint += 1
            if a.apply:
                os.unlink(link)
        else:
            n_new += 1
        if a.apply:
            os.makedirs(d, exist_ok=True)
            os.symlink(text, link)

    mode = "APPLIED" if a.apply else "DRY RUN"
    print(f"[{mode}] target {target}" + ("  (in-project links relative)" if a.relative else ""))
    print(f"  day files found per source (wanted stations only): {seen}")
    print(f"  skipped (station not in {os.path.basename(STATIONS_CSV)}): {n_skip_station}")
    print(f"  links: new {n_new}, already present {n_exist}, re-pointed {n_repoint}")
    if planned:
        df = pd.DataFrame([(k[0], k[1], k[3], k[4], v[1]) for k, v in planned.items()],
                          columns=["net", "sta", "year", "doy", "source"])
        g = df.groupby(["net", "year"]).agg(stations=("sta", "nunique"), days=("doy", "nunique"),
                                            doy_min=("doy", "min"), doy_max=("doy", "max"),
                                            sources=("source", lambda c: "/".join(f"{k}:{v}" for k, v in c.value_counts().items())))
        print(g.to_string())
    if not a.apply:
        print("  (dry run: add --apply to create the links; --check to verify the tree)")


if __name__ == "__main__":
    main()
