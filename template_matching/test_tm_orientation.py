#!/usr/bin/env python3
"""Real-data test of tm_orientation (ISSUES #27): at stations whose orientation (or wiring) differs between the 2020 and
the 2026 sequence, 2020 events are used as templates and 2026 events as targets. For each pair, the S window (5-20 Hz,
S pick -0.5..+1.5 s, the scan's S window) is correlated on N and E (per channel normalised, averaged, as the scan's network
CC; best lag within +-0.3 s), once with the template as recorded and once turned into the target day's frame. The turned
template must match better; a sign error would make it match worse. Pairs within one sequence (no turning) are the
reference. Reads the R4 event exports (raw N/E with the R4 picks).

  python template_matching/test_tm_orientation.py [--pairs 150]
"""
import argparse, glob, os, random
import numpy as np, pandas as pd
from obspy import read

import tm_orientation as TO

W = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "relocation/hypodd_dl_renewal_R4/waveforms_100km")
STATIONS = [("KS", "HANB", "HG"), ("KS", "GOCB", "HH"), ("KS", "HMPA", "HG"), ("KS", "NLDA", "HG"), ("KG", "IBA", "HG"),
            ("KS", "MJDB", "HH"), ("KS", "IJDB", "HH"), ("KS", "JDO2", "EL"), ("KS", "JAGA", "HG"), ("KS", "HAWA", "HG")]
FS, LAG = 100.0, 30


def era(eid):
    return "2020" if eid < "20200801" else ("2026" if eid >= "20260814" else "q")


def load(eid, net, sta, sen):
    out = []
    for c in "NE":
        f = f"{W}/{eid}/{eid}.{net}.{sta}.{sen}{c}.sac"
        if not os.path.exists(f):
            return None
        tr = read(f)[0]; t0 = tr.stats.sac.get("t0", -12345.0)
        if t0 in (None, -12345.0):
            return None
        tr.detrend("demean"); tr.taper(0.05); tr.filter("bandpass", freqmin=5, freqmax=20, corners=4, zerophase=True)
        if abs(tr.stats.sampling_rate - FS) > 1e-6:
            tr.resample(FS)
        i0 = int(round((t0 - tr.stats.sac.b - 0.5) * FS)) - LAG; n = int(2.0 * FS) + 2 * LAG
        x = tr.data[i0:i0 + n].astype(float)
        if i0 < 0 or len(x) < n:
            return None
        out.append(x)
    return np.array(out)                                             # (2, n): N, E with +-LAG margins


def cc2(tpl, tgt):
    """mean over N, E of the per-channel normalised cc; best common lag"""
    T = tpl[:, LAG:-LAG]; T = T - T.mean(1, keepdims=True); nT = np.linalg.norm(T, axis=1)
    best = -2.0
    for k in range(2 * LAG + 1):
        D = tgt[:, k:k + T.shape[1]]; D = D - D.mean(1, keepdims=True); nD = np.linalg.norm(D, axis=1)
        if (nT == 0).any() or (nD == 0).any():
            return np.nan
        best = max(best, float(np.mean(np.sum(T * D, axis=1) / (nT * nD))))
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs", type=int, default=150); a = ap.parse_args()
    random.seed(20261014); rows = []
    for net, sta, sen in STATIONS:
        ev = sorted({os.path.basename(f).split(".")[0] for f in glob.glob(f"{W}/*/*.{net}.{sta}.{sen}N.sac")})
        by = {"2020": [e for e in ev if era(e) == "2020"], "2026": [e for e in ev if era(e) == "2026"]}
        s20 = TO.state(net, sta, sen, "2020-05-01"); s26 = TO.state(net, sta, sen, "2026-08-20")
        cache = {}
        get = lambda e: cache.setdefault(e, load(e, net, sta, sen))
        for cls, A, B in (("2020 -> 2026", "2020", "2026"), ("within 2026", "2026", "2026")):
            res = []
            tries = 0
            while len(res) < a.pairs and tries < 20 * a.pairs and by[A] and by[B]:
                tries += 1
                e1, e2 = random.choice(by[A]), random.choice(by[B])
                if e1 == e2:
                    continue
                x, y = get(e1), get(e2)
                if x is None or y is None:
                    continue
                M = TO.transform(TO.state(net, sta, sen, f"{e1[:4]}-{e1[4:6]}-{e1[6:8]}"), TO.state(net, sta, sen, f"{e2[:4]}-{e2[4:6]}-{e2[6:8]}"))
                if M is None:
                    continue
                res.append((cc2(x, y), cc2(M @ x, y)))
            r = np.array(res)
            if len(r):
                rows.append(dict(station=f"{net}.{sta}.{sen}", pairs=cls, n=len(r), theta_2020=s20[0] if s20 else None, theta_2026=s26[0] if s26 else None,
                                 cc_as_recorded=round(float(np.median(r[:, 0])), 3), cc_turned=round(float(np.median(r[:, 1])), 3),
                                 turned_better=round(float(np.mean(r[:, 1] > r[:, 0] + 1e-9)), 2)))
                print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs/test_tm_orientation.csv"), index=False)
    pd.set_option("display.width", 200); print(out.to_string(index=False))


if __name__ == "__main__":
    main()
