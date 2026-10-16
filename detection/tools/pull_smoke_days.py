# G3 smoke days, pulled ahead of the bulk mirror into the same SDS tree (the bulk rsync then skips them): every chosen stream
# with data that day. Days: 2021-08-27/28 (DOY 239-240), 2026-01-01 (DOY 1), 2013-07-19 (DOY 200, a sparse-network day).
import os, subprocess, pandas as pd
from concurrent.futures import ThreadPoolExecutor
ROOT = "/home/msseo/works/73.Gwangyang_sequence"; LOCAL = f"{ROOT}/data/continuous/KS_KG_mara_sds"
MARA = "<user>@<mara-host>"; SDS = "<mara-sds-root>"
DAYS = [(2021, 239), (2021, 240), (2026, 1), (2013, 200)]
st = pd.read_csv(f"{ROOT}/docs/station_inventory/tables/stations.csv")
jobs = []
for r in st.itertuples():
    for y, d in DAYS:
        day = pd.Timestamp(f"{y}-01-01") + pd.Timedelta(days=d - 1)
        if r.first_day <= str(day.date()) <= r.last_day:
            jobs.append((r.net, r.sta, r.chosen_sensor, y, d))
def run(j):
    net, sta, sen, y, d = j
    dst = f"{LOCAL}/{y}/{net}/{sta}/"; os.makedirs(dst, exist_ok=True)
    inc = [f"--include={sen}{c}.D/" for c in "ZNE"] + [f"--include={sen}{c}.D/{net}.{sta}..{sen}{c}.D.{y}.{d:03d}" for c in "ZNE"]
    r = subprocess.run(["rsync", "-aL", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=20", *inc, "--exclude=*", f"{MARA}:{SDS}/{y}/{net}/{sta}/", dst],
                       capture_output=True, text=True)
    return r.returncode
with ThreadPoolExecutor(8) as ex:
    rc = list(ex.map(run, jobs))
print(f"smoke-day pulls: {len(jobs)} station-days, rc counts {pd.Series(rc).value_counts().to_dict()}")
