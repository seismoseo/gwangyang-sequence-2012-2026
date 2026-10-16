#!/usr/bin/env python
# GWANGYANG addition (Haenam R3x rule, 2026-10-13): a predecessor code ends the day before its successor at the same site starts,
# so no site is counted twice. Same site = within 1 km (co-sited pairs listed by the 2026-10-15 check); the co-located second
# broadband KWJU stays beside KWJ2 for its whole period (Haenam renewal decision B: two instruments, both used).
# Writes the LinkUntil column of detection/station_table/stations.csv (read by detection/tools/link_continuous.py).
#   python detection/station_table/add_link_until.py            # dry run
#   python detection/station_table/add_link_until.py --apply
import math, sys, os
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
INV = os.path.join(HERE, "../../docs/station_inventory/tables/stations.csv")
OUT = os.path.join(HERE, "stations.csv")
SAME_SITE_KM = 1.0
BOTH_USED = {("KWJU", "KWJ2"), ("KWJ2", "KWJU")}
s = pd.read_csv(INV); t = pd.read_csv(OUT)
until = {}
for i, a in s.iterrows():
    for j, b in s.iterrows():
        if j == i or a.role.startswith("temporary") or b.role.startswith("temporary") or (a.sta, b.sta) in BOTH_USED:
            continue
        d = 111.19 * math.hypot(a.lat - b.lat, (a.lon - b.lon) * math.cos(math.radians(a.lat)))
        if d <= SAME_SITE_KM and a.first_day < b.first_day <= a.last_day:          # a is older and still running when b starts
            u = str((pd.Timestamp(b.first_day) - pd.Timedelta(days=1)).date())
            if (a.net, a.sta) not in until or u < until[(a.net, a.sta)][0]:
                until[(a.net, a.sta)] = (u, b.sta, round(d * 1000))
t["LinkUntil"] = [until.get((n, c), ("",))[0] for n, c in zip(t.Network, t.Code)]
for (n, c), (u, succ, dm) in sorted(until.items()):
    print(f"{n}.{c}: linked until {u} (successor {succ} starts the next day, {dm} m away)")
if "--apply" in sys.argv:
    t.to_csv(OUT, index=False); print(f"wrote LinkUntil into {OUT}")
else:
    print("DRY RUN (add --apply)")
