"""Template N/E in the orientation of the scanned day (renewal R5; ISSUES #27, the user's design of 2026-10-12).

P on Z and S on N/E at every station. The continuous data are never rotated. Where a sensor's orientation (or its wiring
repair) on the template's day differs from the scanned day, the template's horizontals are turned into the scanned day's
frame from the one orientation table (registry/orientation.py: theta and repair, the same functions every ZRT product uses):

    healthy = Rep(day) . recorded             (Rep: the reversed channel x -1 on the days its fault certainly holds)
    healthy = R(theta(day)) . geographic       R(a) = [[cos a, sin a], [-sin a, cos a]]  (sensor N axis at azimuth a)
    => recorded(scan day D) = Rep(D) . R(theta_D - theta_T) . Rep(T) . recorded(template day T)

A template channel pair needs both N and E to be turned; where only one exists, or theta is unknown on either day, the
station's S is dropped for that template on that day (never scanned unturned).
"""
import pathlib, sys
from functools import lru_cache

import numpy as np

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry"))
import orientation as O   # noqa: E402

TOL = 1e-9


@lru_cache(maxsize=None)
def state(net, sta, sensor, day):
    """(theta_deg, sign_N, sign_E) on a day (YYYY-MM-DD), or None when the orientation is unknown"""
    th, _ = O.theta(net, sta, sensor, day)
    if th is None:
        return None
    n, e, _ = O.repair(np.array([1.0]), np.array([1.0]), sta, sensor, day)
    return float(th), float(n[0]), float(e[0])


def transform(s_tpl, s_day):
    """2x2 matrix taking a template's recorded (N, E) into the scanned day's recorded frame; None if either state is unknown"""
    if s_tpl is None or s_day is None:
        return None
    a = np.radians(s_day[0] - s_tpl[0])
    R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    return np.diag([s_day[1], s_day[2]]) @ R @ np.diag([s_tpl[1], s_tpl[2]])


def is_identity(M):
    return M is not None and np.allclose(M, np.eye(2), atol=TOL)


def apply_to_day(S_waveforms, S_weights, stations, prefixes, origin_utc, s_comp, day, log=print, day_stations=None):
    """Turn every template's S (N, E) into the frame of `day` (YYYY-MM-DD) at the horizontal stations.
    day_stations: per station, the code whose data the scan reads that day (scan.day_codes; a temporary code such as
    KS.JAEBT in place of JAEB has its own orientation); default the template's station codes.
    Returns (S_waveforms for this day, S_weights with the untransformable pairs zeroed, counts). The input waveform array is
    not modified; a copy is made only when some template needs turning."""
    nT = S_waveforms.shape[0]
    W = S_waveforms; Sw = S_weights; copied = False
    n_turned = n_dropped = 0
    tdays = [str(o)[:10] for o in origin_utc]
    for si, key in enumerate(stations):
        if s_comp[si] == "Z":
            continue
        net, sta = str(key).split(".")
        sensor = str(prefixes[si])
        sd = state(net, day_stations[si] if day_stations is not None else sta, sensor, day)
        for ti in range(nT):
            if not (Sw[ti, si] > 0).any():
                continue
            M = transform(state(net, sta, sensor, tdays[ti]), sd)
            if is_identity(M):
                continue
            has = np.abs(W[ti, si]).max(axis=-1) > 0
            if M is None or not has.all():
                Sw[ti, si, :] = 0.0; n_dropped += 1
                continue
            if not copied:
                W = S_waveforms.copy(); copied = True
            W[ti, si] = (M @ W[ti, si].astype(np.float64)).astype(np.float32)
            n_turned += 1
    return W, Sw, dict(turned=n_turned, dropped=n_dropped)
