#!/usr/bin/env python3
"""Recover station-days whose day files contain damaged miniSEED records (2021-03-03: the same damage in the MARA archive
and in NECIS). Each record (block) is decoded on its own; a record that fails to decode, or decodes with a Steim2
integrity warning, is dropped; every other record is kept byte for byte. The day then has short gaps where records were
dropped. User decision 2026-10-13.

For each station and day given, all three components of the station's chosen sensor are processed (a component without
damage is copied unchanged). Output: data/continuous/repaired_20261013/<YYYY>/<STA>/<NET>.<STA>.<CHA>.<YYYY>.<DDD>.00.00.00
(the naming the link tree reads as a source) and repaired_20261013/repair_log.csv: records kept/dropped, samples kept,
and a check that every kept sample equals the source wherever the source itself can be read.

  python tools/repair_mseed_records.py --day 2021.062 --stations CJD,CSDB,... [--apply]
"""
import argparse, io, os, warnings
import numpy as np, pandas as pd
from obspy import read
from obspy.io.mseed.util import get_record_information

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARA = os.path.join(os.path.dirname(HERE), "data/continuous/KS_KG_mara_sds")
OUT = os.path.join(os.path.dirname(HERE), "data/continuous/repaired_20261013")
ST = pd.read_csv(os.path.join(HERE, "station_table/stations.csv"))


def records(path):
    """yield (offset, length, bytes) of every record of a miniSEED file"""
    data = open(path, "rb").read(); off = 0
    while off < len(data):
        info = get_record_information(path, offset=off)
        n = info["record_length"]
        yield off, n, data[off:off + n]
        off += n


def decode_ok(rec):
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # an integrity warning counts as damage
        try:
            st = read(io.BytesIO(rec), format="MSEED")
            return True, sum(t.stats.npts for t in st)
        except Exception:
            return False, 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--day", required=True); ap.add_argument("--stations", required=True)
    ap.add_argument("--apply", action="store_true"); a = ap.parse_args()
    y, d = a.day.split(".")
    rows = []
    for sta in a.stations.split(","):
        r0 = ST[ST.Code == sta].iloc[0]; net, sen = r0.Network, r0.Sensor
        for c in "ZNE":
            cha = f"{sen}{c}"; src = os.path.join(MARA, y, net, sta, f"{cha}.D", f"{net}.{sta}..{cha}.D.{y}.{d}")
            if not os.path.exists(src):
                rows.append(dict(sta=sta, cha=cha, note="no source file")); continue
            kept, dropped, nkept = [], 0, 0
            for off, n, rec in records(src):
                ok, ns = decode_ok(rec)
                if ok:
                    kept.append(rec); nkept += ns
                else:
                    dropped += 1
            blob = b"".join(kept)
            # check: the repaired file reads, and equals the source wherever the source itself can be read
            st_rep = read(io.BytesIO(blob), format="MSEED")
            for t in st_rep:                                      # the pipeline's own rule (core.py): rates within 1e-3 of an integer are rounded
                if 0 < abs(t.stats.sampling_rate - round(t.stats.sampling_rate)) < 1e-3:
                    t.stats.sampling_rate = float(round(t.stats.sampling_rate))
            st_rep.merge(method=1)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore"); src_st = read(src)
                same = tot = 0
                for tr in src_st:
                    t = st_rep[0]; i0 = int(round((tr.stats.starttime - t.stats.starttime) * t.stats.sampling_rate))
                    if i0 < 0:
                        continue
                    seg = np.ma.filled(t.data[i0:i0 + tr.stats.npts], -2**31); k = min(len(seg), tr.stats.npts)
                    same += int(np.sum(seg[:k] == tr.data[:k])); tot += k
                check = f"{same}/{tot} identical"
            except Exception:
                check = "source unreadable"
            rows.append(dict(sta=sta, cha=cha, records=len(kept) + dropped, dropped=dropped, samples_kept=nkept,
                             day_fraction=round(nkept / 8640000, 4), check_vs_source=check))
            if a.apply:
                dst = os.path.join(OUT, y, sta, f"{net}.{sta}.{cha}.{y}.{d}.00.00.00")
                os.makedirs(os.path.dirname(dst), exist_ok=True); open(dst, "wb").write(blob)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200); print(df.to_string(index=False))
    if a.apply:
        os.makedirs(OUT, exist_ok=True)
        lp = os.path.join(OUT, "repair_log.csv")
        df.assign(day=a.day).to_csv(lp, mode="a", header=not os.path.exists(lp), index=False)
        print("written to", OUT)


if __name__ == "__main__":
    main()
