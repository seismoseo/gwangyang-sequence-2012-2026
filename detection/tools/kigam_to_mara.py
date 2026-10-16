#!/usr/bin/env python
"""Deliver the KIGAM-fetched KG waveforms to MARA so the stp server serves them like every other record.

MARA's stp server reads one SDS symlink tree; new data reach it in two steps (MARA's
/home/msseo/0.server_management/CLAUDE.md): (1) the files go into an msseo-owned archive folder
/data/backup/nfs/anystor/<NN>.<NET>.<YYYYMMDD>; (2) a root-run linker adds their links to the tree.
This tool does step 1 and installs the linker for step 2; you run the linker once with sudo on MARA.

  stage   reflink copies (no extra disk) of data/continuous/KG_kigam_fdsn_sds into STAGE in MARA's day-major
          layout, the one 41.KG.20241111 uses:  <YYYY>/<JJJ>/KG.<STA>.<CHA>.<YYYY>.<JJJ>.<HH>.<MM>.<SS>
          (HH.MM.SS = the file's first sample); writes README_KIGAM_FILL.txt with the provenance
  push    rsync STAGE to MARA (new folder: a plain archive copy), then copies the linker and its apply script
          into MARA's server_management folder
Both are DRY RUN by default (--apply to act), idempotent and safe to re-run while the KIGAM fetch is still
running: only finished day files are staged (the fetch writes <file>.part and renames it when complete).

  python detection/tools/kigam_to_mara.py stage [--apply]
  python detection/tools/kigam_to_mara.py push  [--apply]
then on MARA:  sudo bash /home/msseo/0.server_management/apply_KG50.sh
"""
import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # the project registry, found upward from this file
import argparse
import glob
import os
import subprocess

# ---------------------------------------------------------------- parameters (edit here)
SRC = str(haenam.ROOT / "data/continuous/KG_kigam_fdsn_sds")
ARCHIVE = "50.KG.20261011"                                  # MARA naming: <global counter>.<NET>.<YYYYMMDD>
STAGE = f"/home/msseo/works/_mara_staging/{ARCHIVE}"
MARA = "<user>@<mara-host>"
MARA_DEST = f"/data/backup/nfs/anystor/{ARCHIVE}"
MARA_MGMT = "/home/msseo/0.server_management"
LINKER_FILES = [str(haenam.ROOT / "detection/tools/mara/make_symbolic_link_KG50.py"),
                str(haenam.ROOT / "detection/tools/mara/apply_KG50.sh")]
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]


_FIRST = None


def first_sample(net, sta, cha, yr, jd):
    """first-sample time of one fetched day file, from the fetch log (no file read; the disk is busy with the fetch)"""
    global _FIRST
    if _FIRST is None:
        import csv, datetime as dt
        _FIRST = {}
        for r in csv.DictReader(open(f"{SRC}/_manifest/fetch_log.csv")):
            if r["status"] == "ok":
                d = dt.date.fromisoformat(r["day"]); j = d.timetuple().tm_yday
                _FIRST[(r["station"], r["channel"], str(d.year), f"{j:03d}")] = r["first"]
    s = _FIRST.get((sta, cha, yr, jd))
    if s is None:                                     # not in the log: read the first record
        from obspy.io.mseed.util import get_record_information
        t = get_record_information(f"{SRC}/{yr}/{net}/{sta}/{cha}.D/{net}.{sta}..{cha}.D.{yr}.{jd}")["starttime"]
        return t.hour, t.minute, t.second
    hms = s[11:19].split(":")                         # e.g. 2024-11-01T00:00:00.000000Z
    return int(hms[0]), int(hms[1]), int(hms[2])


def staged_name(src):
    """KG.<STA>..<CHA>.D.<YYYY>.<JJJ> -> (<YYYY>/<JJJ>, KG.<STA>.<CHA>.<YYYY>.<JJJ>.<HH>.<MM>.<SS>)"""
    b = os.path.basename(src).split(".")              # KG STA '' CHA D YYYY JJJ
    net, sta, cha, yr, jd = b[0], b[1], b[3], b[5], b[6]
    h, m, sec = first_sample(net, sta, cha, yr, jd)
    return f"{yr}/{jd}", f"{net}.{sta}.{cha}.{yr}.{jd}.{h:02d}.{m:02d}.{sec:02d}"


def cmd_stage(a):
    files = sorted(f for f in glob.glob(f"{SRC}/[12][0-9][0-9][0-9]/KG/*/*.D/KG.*") if not f.endswith(".part"))
    todo = []
    for f in files:
        sub, name = staged_name(f)
        dst = os.path.join(STAGE, sub, name)
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(f):
            continue
        todo.append((f, dst))
    print(f"stage: {len(files):,} finished day files in {SRC}; {len(todo):,} not yet staged -> {STAGE}")
    if not a.apply:
        for f, d in todo[:3]:
            print(f"  cp --reflink=always {f} {d}")
        print("DRY RUN: nothing copied (add --apply)"); return 0
    for f, d in todo:
        os.makedirs(os.path.dirname(d), exist_ok=True)
        subprocess.run(["cp", "-p", "--reflink=always", f, d], check=True)
    stas = sorted({os.path.basename(f).split(".")[1] for f in files})
    chans = sorted({os.path.basename(f).split(".")[3] for f in files})
    yrs = sorted({os.path.basename(f).split(".")[5] for f in files})
    with open(os.path.join(STAGE, "README_KIGAM_FILL.txt"), "w") as fh:
        fh.write(f"""{ARCHIVE}: KG continuous waveforms fetched from the KIGAM Quake FDSN service
(https://quake.kigam.re.kr/, fdsnws/dataselect) to fill the KG gap after 41.KG.20241111 (which ends 2024.305).

Fetched 2026-10-11 by the Haenam project (27.Haenam_sequence/detection/tools/fetch_kigam_fdsn.py).
Stations : {', '.join(stas)}   (the KG stations within 100 km of the Haenam source; NOT the whole KG network)
Channels : {', '.join(chans)}   (100-sps sensors only; the 20-sps BG/BH/SL copies were not fetched)
Period   : 2024-11-01 .. 2026-09-30 (years {', '.join(yrs)}); one file per channel and UTC day, trimmed to the day
Layout   : day-major like 41.KG: <YYYY>/<JJJ>/KG.<STA>.<CHA>.<YYYY>.<JJJ>.<HH>.<MM>.<SS> (HH.MM.SS = first sample)
Check    : 36 of 36 channel-days of 2024-10-29/30 fetched from KIGAM are sample-for-sample identical to 41.KG.
Linking  : make_symbolic_link_KG50.py (apply_KG50.sh, run as root).
""")
    print(f"staged {len(todo):,} files; README written"); return 0


def cmd_push(a):
    rs = ["rsync", "-a", "--partial", "--exclude=*.part", "-e", " ".join(SSH), STAGE + "/", f"{MARA}:{MARA_DEST}/"]
    print("push:", " ".join(rs if a.apply else rs[:1] + ["--dry-run", "--stats"] + rs[1:]))
    if not a.apply:
        r = subprocess.run(rs[:1] + ["--dry-run", "--stats"] + rs[1:], capture_output=True, text=True)
        print("\n".join(l for l in r.stdout.splitlines() if l.startswith(("Number of", "Total file size", "Total transferred"))))
        print("DRY RUN: nothing sent (add --apply)"); return r.returncode
    subprocess.run(SSH + [MARA, f"mkdir -p {MARA_DEST}"], check=True)
    subprocess.run(rs, check=True)
    subprocess.run(["scp", "-o", "BatchMode=yes"] + LINKER_FILES + [f"{MARA}:{MARA_MGMT}/"], check=True)
    print(f"pushed to {MARA}:{MARA_DEST} and installed {', '.join(os.path.basename(x) for x in LINKER_FILES)}")
    print(f"next (on MARA, as root): sudo bash {MARA_MGMT}/apply_KG50.sh"); return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("stage", "push"):
        sub.add_parser(n).add_argument("--apply", action="store_true")
    a = ap.parse_args()
    return {"stage": cmd_stage, "push": cmd_push}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
