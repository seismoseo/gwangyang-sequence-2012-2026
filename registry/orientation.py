"""The one orientation table and the one rotation function of the renewal (rule "Rotation", ISSUES #25 and #29).

Every ZRT product (S differential times on T, ASTF/SDM/slip, P polarity) turns N/E into R/T through this module, so
all of them use the same angles, the same repairs and the same back azimuth.

  theta(net, sta, sensor, t)        the misorientation theta_M (degrees) valid at time t, or None where it is unknown
  rotate_rt(n, e, net, sta, sensor, t, stla, stlo)
                                     -> (r, t, info): N/E repaired, then rotated to R/T about the shared back-azimuth
                                        point; None if the orientation at t is unknown

Rules (ISSUES.md "Rules in force", #25, #29; the user's decisions of 2026-10-11/12):
- angles come from the registry product orientation_periods (R1c table): one theta_M per sensor and period;
- one shared back-azimuth point for every event: BAZ_POINT (the median of the 25 m reference set);
- R/T as in the project's ZRT notebooks: ba = (baz - theta_M) % 360, then obspy's rotate_ne_rt;
- a period runs to the day before the next period starts when the change is dated (consecutive periods); the first
  period is extended back to the window start and the last forward to the window end (measurements start after, and end
  before, the data);
- between two periods that do not touch (a change bracket, #25): the earlier angle persists up to the day before the
  first measurement of the new angle (user's decision, 2026-10-13); UNKNOWN_IN_BRACKETS = True would return None there;
- a sensor never measured -> 0 deg (nominal north), flagged "not measured";
- wiring faults (#29), from 33's flip verdicts: the reversed channel is multiplied by -1 before rotating. A whole-record
  fault holds on every day (with 33's theta after repair where the table has no period). An interval fault holds from
  the first to the last measurement that is really unhealthy (mirrored) (user's decision, 2026-10-13); outside that
  span the channel is healthy. BGDB.HG: 2020-09-12 .. 2021-05-21. (33's table interval 2020-08-05..2021-07-13 is the
  span it searched; it starts and ends on healthy measurements.)
- the table (orientation_periods) already carries both dating rules for orientation changes: an instrument epoch that
  falls alone inside a bracket dates the change; otherwise the earlier angle holds up to the day before the first
  measurement of the new angle (user's decisions, 2026-10-13). measured_start/measured_end keep the measurements.
"""
import os
from functools import lru_cache

import numpy as np
import pandas as pd

import haenam

BAZ_POINT = (35.0000, 127.7286)                 # GWANGYANG PORT: the shared back-azimuth point = the Gwangyang centroid
WINDOW = ("2012-12-20", "2026-10-06")           # GWANGYANG PORT: the Gwangyang study window (decision P1)
UNKNOWN_IN_BRACKETS = False                     # user, 2026-10-13: the earlier angle persists
FLIPS_33 = "/home/msseo/works/33.KS_station_orientation/national/Table/flip_twin_verdicts.csv"


@lru_cache(maxsize=1)
def periods():
    o = pd.read_csv(haenam.path(None, "orientation_periods"))
    o["start"] = pd.to_datetime(o.start); o["end"] = pd.to_datetime(o.end)
    return {k: g.sort_values("start").reset_index(drop=True) for k, g in o.groupby(["net", "sta", "sensor"])}


@lru_cache(maxsize=1)
def faults():
    """{(sta, sensor): dict(channel, certain=(t0, t1), unknown=[(a, b), ...], theta_repaired)} from 33's flip verdicts"""
    out = {}
    if not os.path.exists(FLIPS_33):
        return out
    for r in pd.read_csv(FLIPS_33).itertuples():
        sta, sensor = r.unit.split(".")
        ch = "E" if r.kind.startswith("E") else "N" if r.kind.startswith("N") else None
        if ch is None:
            continue
        if r.extent == "whole record":
            certain, unknown = (pd.Timestamp("1900-01-01"), pd.Timestamp("2100-01-01")), []
        else:
            certain, unknown = (pd.Timestamp(r.first_mirrored), pd.Timestamp(r.last_mirrored)), []   # first..last mirrored (user)
        out[(sta, sensor)] = dict(channel=ch, certain=certain, unknown=unknown, theta_repaired=float(r.theta_repaired), extent=r.extent)
    return out


def _day(t):
    return pd.Timestamp(str(t)[:10])


def theta(net, sta, sensor, t):
    """(theta_M in degrees or None, note)"""
    d = _day(t)
    f = faults().get((sta, sensor))
    if f and any(a <= d <= b for a, b in f["unknown"]):
        return None, "wiring-fault bracket: unknown"
    g = periods().get((net, sta, sensor))
    if g is None or not len(g):
        if f and f["extent"] == "whole record":
            return f["theta_repaired"], "whole-record wiring fault: 33 theta after repair"
        return 0.0, "not measured: 0 deg"
    w0, w1 = pd.Timestamp(WINDOW[0]), pd.Timestamp(WINDOW[1])
    for i, r in g.iterrows():
        s = w0 if i == 0 and r.start > w0 else r.start
        e = w1 if i == len(g) - 1 and r.end < w1 else r.end
        if i + 1 < len(g) and g.start[i + 1] == r.end + pd.Timedelta(days=1):
            e = r.end                                                   # dated change: consecutive periods
        if s <= d <= e:
            return float(r.theta_M_apply), f"period {r.start.date()}..{r.end.date()}"
        if i + 1 < len(g) and r.end < d < g.start[i + 1]:
            if UNKNOWN_IN_BRACKETS:
                return None, f"change bracket {r.end.date()}..{g.start[i + 1].date()}: unknown"
            return float(r.theta_M_apply), f"change bracket {r.end.date()}..{g.start[i + 1].date()}: earlier angle kept"
    return None, "outside the table"


def repair(n, e, sta, sensor, t):
    """N/E with a reversed channel multiplied by -1 on the days the fault certainly holds; (n, e, note)"""
    f = faults().get((sta, sensor))
    if not f:
        return n, e, ""
    d = _day(t)
    if f["certain"][0] <= d <= f["certain"][1]:
        return (-n, e, "N reversed: repaired") if f["channel"] == "N" else (n, -e, "E reversed: repaired")
    return n, e, ""


def back_azimuth(stla, stlo):
    from obspy.geodetics import gps2dist_azimuth
    return gps2dist_azimuth(BAZ_POINT[0], BAZ_POINT[1], stla, stlo)[2]


def rotate_rt(n, e, net, sta, sensor, t, stla, stlo):
    """(r, t, info) or None. n, e: equal-length arrays of the sensor's raw N and E; t: the event's origin time."""
    from obspy.signal.rotate import rotate_ne_rt
    th, note = theta(net, sta, sensor, t)
    if th is None:
        return None
    n, e, rep = repair(np.asarray(n, float), np.asarray(e, float), sta, sensor, t)
    baz = back_azimuth(stla, stlo)
    ba = (baz - th) % 360
    r, tt = rotate_ne_rt(n, e, ba)
    return r, tt, dict(theta=th, baz=baz, ba=ba, note="; ".join(x for x in (note, rep) if x))
