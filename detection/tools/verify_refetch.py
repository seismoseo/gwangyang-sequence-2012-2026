#!/usr/bin/env python3
"""Check the NECIS re-fetched day files (data/continuous/necis_refetch_20261013) before the link tree uses them.
For every re-fetched file of a station's CHOSEN sensor (the one detection reads):
  1. it reads without error;
  2. it is one day of data: at most 100 fragments and no more than 0.1% over 8,640,000 samples (100 Hz), e.g. overlapping fragments;
  3. wherever the MARA copy can be read, the samples are identical (the same recording).
A file that fails 1 or 3 is moved to rejected/ so the link tree never takes it. Writes verify_refetch.csv beside the files.

  python tools/verify_refetch.py [--apply]     (without --apply: report only, nothing moved)
"""
import argparse, glob, os, shutil, warnings
import numpy as np, pandas as pd
from obspy import read
warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(os.path.dirname(HERE), "data/continuous/necis_refetch_20261013")
MARA = os.path.join(os.path.dirname(HERE), "data/continuous/KS_KG_mara_sds")
ST = pd.read_csv(os.path.join(HERE, "station_table/stations.csv"))
SENSOR = dict(zip(ST.Code, ST.Sensor))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true"); a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(REF, "continuous/*/*/*"))):
        net, sta, cha, y, d = os.path.basename(f).split(".")[:5]
        if SENSOR.get(sta) != cha[:2]:
            continue                                            # not the sensor detection reads
        r = dict(file=os.path.basename(f), sta=sta, cha=cha, day=f"{y}.{d}")
        try:
            n = read(f); r["necis_traces"] = len(n); r["necis_samples"] = sum(t.stats.npts for t in n); r["necis_read"] = "ok"
        except Exception as e:
            r["necis_read"] = f"ERROR {str(e)[:60]}"; rows.append(r); continue
        mf = os.path.join(MARA, y, net, sta, f"{cha}.D", f"{net}.{sta}..{cha}.D.{y}.{d}")
        try:
            m = read(mf); r["mara_read"] = "ok"; r["mara_samples"] = sum(t.stats.npts for t in m)
            n.merge(method=1); nt = n[0]; same = tot = 0
            for tr in m:
                i0 = int(round((tr.stats.starttime - nt.stats.starttime) * nt.stats.sampling_rate))
                if i0 < 0:
                    continue
                seg = np.ma.filled(nt.data[i0:i0 + tr.stats.npts], -2**31); k = min(len(seg), tr.stats.npts)
                same += int(np.sum(seg[:k] == tr.data[:k])); tot += k
            r["overlap_identical"] = same; r["overlap_total"] = tot
        except FileNotFoundError:
            r["mara_read"] = "absent"
        except Exception as e:
            r["mara_read"] = f"unreadable ({str(e)[-40:].strip()})"
        tot = r.get("overlap_total", 0)
        r["verdict"] = ("REJECT: NECIS unreadable" if r["necis_read"] != "ok" else
                        "REJECT: not one clean day (traces/samples)" if r["necis_traces"] > 100 or r["necis_samples"] > 8640000 * 1.001 else
                        "REJECT: differs from MARA" if tot and r["overlap_identical"] < 0.999 * tot else "use")
        rows.append(r)
    df = pd.DataFrame(rows)
    # a station-day is used only if ALL its components pass (detection reads Z, N and E together); else all stay out
    sd_ok = df.groupby(["sta", "day"]).verdict.apply(lambda v: (v == "use").all() and len(v) == 3)
    df["station_day"] = [("use" if sd_ok[(r.sta, r.day)] else "REJECT (a component fails)") for r in df.itertuples()]
    df.loc[df.station_day != "use", "verdict"] = df.loc[df.station_day != "use", "verdict"].where(df.verdict != "use", "REJECT: another component fails")
    df.to_csv(os.path.join(REF, "verify_refetch.csv"), index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
    print(df[[c for c in ["day", "sta", "cha", "necis_traces", "necis_samples", "mara_read", "mara_samples", "overlap_identical", "overlap_total", "verdict"] if c in df]].to_string(index=False))
    rej = df[df.verdict != "use"]
    sd = df.drop_duplicates(["sta", "day"])
    print(f"\n{len(df)} files of chosen sensors: use {int((df.verdict == 'use').sum())}, reject {len(rej)}; "
          f"station-days usable: {int((sd.station_day == 'use').sum())} of {len(sd)}")
    print(sd.groupby("day").station_day.apply(lambda v: f"{(v == 'use').sum()}/{len(v)} usable").to_string())
    if a.apply and len(rej):
        os.makedirs(os.path.join(REF, "rejected"), exist_ok=True)
        for r in rej.itertuples():
            src = glob.glob(os.path.join(REF, "continuous/*/*", r.file))
            for s in src:
                shutil.move(s, os.path.join(REF, "rejected", r.file))
        print(f"moved {len(rej)} rejected files to {os.path.join(REF, 'rejected')}")


if __name__ == "__main__":
    main()
