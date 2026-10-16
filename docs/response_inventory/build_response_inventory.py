#!/usr/bin/env python
"""Response inventory of the Haenam renewal (step R1d, now part of R6 magnitudes) and the check of ISSUES #20.

What it does
  One merged instrument-response inventory for every CHOSEN sensor (one sensor per station: the Sensor column of
  detection/station_table/stations.csv, which is what the continuous link layer data/continuous/sds_links holds),
  Z/N/E channels, over the detection window 2018-01-31 .. 2026-09-30, and a day-by-day check that every chosen channel
  has a response on every day it has data. It also compares every NECIS RESP snapshot found on disk (per-event
  packages and station fetches) with what the four primary sources say for the snapshot day (ISSUES #20).

How to run (it writes ONLY into its own folder; every source is read only)
    nice -n 15 /home/msseo/miniforge3/bin/python \
        /home/msseo/works/27.Haenam_sequence_renewal/docs/decisions/response_inventory_20261014/build_response_inventory.py
  About 5 min; peak memory about 4 GB (the 148 MB master StationXML). One process, no GPU.
  Re-run it when a source changes, e.g. after NECIS event packages of the two sequences are downloaded into
  /home/msseo/works/Claude/data/necis/events/<UTC>/RESP/<NECIS_ID>.r/ (SNAPSHOTS picks them up).

Sources. The readers are those of ../station_inventory_20261011/build_station_inventory.py (response_epochs(): the same
files read with obspy.read_inventory, the latest KIGAM file), extended from Z to Z/N/E and kept with their full
responses; they read the renewal tree's copies, which are byte-identical to the old-tree files that builder names.
  master XML       data/stations/responses/master/KS_KG_metadata_1.0.2.xml    KS+KG history; created 2024-06-26
  NECIS current    data/stations/responses/necis_fetched_2026*/extracted/RESP.*  current epoch per KS channel; 2026-10-11
  KIGAM            data/stations/responses/master/KIGAM_KG_<date>.xml (latest)  KG, KIGAM FDSN; 2026-10-11
  JU StationXML    data/jnu/JU_station.xml                                      JU, merged RESP; created 2025-12-15
  NECIS snapshots  NECIS RESP files elsewhere on disk (SNAPSHOTS): each shows the epoch in force on its own date

Rules (README.md states them with their effect; those marked PROPOSAL wait for the user's approval)
  Confirmed vs extrapolated. A source epoch states the response from its start up to the end of its source's issue day
  (master 2024-06-26, NECIS current 2026-10-11, KIGAM 2026-10-11, JU 2025-12-15, a snapshot the day of its event or
  fetch), or up to its own end if earlier: that span is CONFIRMED. After the issue day an open epoch is only carried
  forward (EXTRAPOLATED); an epoch missing from every source (ISSUES #20) can hide only there.
  At every instant the merged response is chosen as follows:
  M1  Among confirmed primary epochs, the master XML is the history (epochs as written, start and end read literally).
  M2  ... except that a NECIS-current (KS) or KIGAM (KG) epoch starting LATER than every master epoch of the channel
      replaces the master from its start on; another NECIS-current/KIGAM epoch is used only where no master epoch is
      confirmed (M3). JU: the JU StationXML is the only source.
  M4  (PROPOSAL P1, USE_SNAPSHOTS; OFF by default) where no primary epoch is confirmed, a confirmed NECIS snapshot
      epoch is used (the one with the latest start). Snapshots never replace a confirmed primary epoch.
  M5  Where nothing is confirmed, the open epoch of the MOST RECENT statement (latest issue date; snapshots only under
      USE_SNAPSHOTS) is carried forward. Such days are counted as "unconfirmed".
  N   NECIS revising itself: a NECIS statement (snapshot or current) is dropped where a later-issued NECIS statement
      describes the same time differently (e.g. the 2023 Centaur gains NECIS corrected by 2024).
  X   (conservative default) EXCLUDE: spans where the waveform checks (waveform_checks.py) show that no documented
      response fits the data carry no response at all; the proposals in README.md would replace them.
  Continuity (part of M1): a master epoch still open at the master's issue day stays confirmed while a later source shows
  an epoch that began no later than START_TOL_DAYS after it and is still running (no instrument change since); the
  master's description is kept and any disagreement with the later description is a conflict, not a new epoch. M2's
  "later" likewise means more than START_TOL_DAYS later.
  Coverage: a day counts as covered when the merged epochs cover all of it. End dates are read literally: an epoch that
  ends at 00:00:00 of day X does not cover day X; days left open only by that are "boundary-day holes" (reading such an
  end as "through day X" is proposal P2 in README.md, NOT applied).
  Responses are compared as the ANALOG sensitivity |H_a(f)| at SENS_FREQS: the product of every stage gain, with the
  poles-zeros stages evaluated at f and the digital FIR stages taken at their gain only. Why: the sources state their
  InstrumentSensitivity at different frequencies (KIGAM 0.05 Hz, master 1 or 5 Hz, NECIS 1 Hz), and the NECIS RESP files
  carry no FIR stages, whose pass-band ripple alone (up to about 1 % at 5 Hz for a Q330HRS) would otherwise look like a
  disagreement. Two responses agree when |H_a| agrees within SENS_TOL at every frequency of SENS_FREQS.
  Conflict day: two confirmed statements covering the day (or the merged response and one) disagree; opposite
  polarity (analog responses more than 90 degrees apart in phase at SENS_FREQS[-1]) is counted apart (sign conflict).
  Suspect epoch: a poles-zeros stage of the sensor with no poles (a flat sensor response), which no seismometer has.

Writes (this folder)
  merged_inventory.xml   StationXML: chosen sensors only, Z/N/E, merged epochs with full responses; every channel epoch
                         carries a comment naming its source, file, source epoch and merge rule
  epochs_chosen.csv      one row per chosen channel and merged epoch
  source_epochs.csv      every source epoch of the chosen channels (snapshots de-duplicated), with what it states
  coverage.csv           one row per chosen stream: data days, covered, gaps, unconfirmed, conflicts (with day ranges)
  issue20_necis_resp.csv one row per NECIS snapshot RESP file of a chosen channel, against the primary sources and the
                         merged inventory on the snapshot day
  build_log.txt          this run's printed log (reader cross-check, counts, the remove_response validation)
"""
import copy
import glob
import io
import os
import re
import sys
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------------------------------- parameters
ROOT = Path("/home/msseo/works/73.Gwangyang_sequence")          # GWANGYANG PORT
HERE = Path(__file__).resolve().parent
W0, W1 = "2012-12-20", "2026-10-06"           # GWANGYANG PORT: the study window
MAIN0 = "2012-12-20"                          # GWANGYANG PORT: no separate main window
SEQ = {"2020": ("2021-07-01", "2021-09-30"), "2026": ("2026-01-01", "2026-02-28")}   # GWANGYANG PORT: the 2021 and 2026 episodes (keys kept for the copied column names)
STATION_TABLE = ROOT / "detection/station_table/stations.csv"     # chosen sensor per code (Sensor), incl. the 2018-19 codes
REGISTRY_STREAMS = ROOT / "registry/streams.csv"                   # cross-check only (the 82 codes of 2020-2026)
LINK_TREE = ROOT / "data/continuous/sds_links"                     # NET/STA/CHA.D/NET.STA..CHA.D.YYYY.DOY
RESP = ROOT / "data/stations/responses"
MASTER_XML = RESP / "master/KS_KG_metadata_1.0.2.xml"
NECIS_CURRENT_GLOB = str(RESP / "necis_fetched_2026*/extracted/RESP.*")
NECIS_CURRENT_ISSUED = "2026-10-11"                                # fetch day (necis_fetched_20261011/README.md)
KIGAM_XML_GLOB = str(RESP / "master/KIGAM_KG_*.xml")               # the latest is used, as in the station-inventory builder
JU_XML = Path("/home/msseo/works/27.Haenam_sequence/data/jnu/JU_station.xml")   # GWANGYANG PORT: unused (no JU stations)
OLD_EPOCH_TABLE = ROOT / "docs/station_inventory/tables/response_epochs.csv"   # GWANGYANG PORT: reader cross-check
NECIS_CATALOG = "/home/msseo/works/Claude/data/necis/catalog/necis_events_domestic_1978_2026.csv"  # NECIS id -> origin UTC
# NECIS RESP snapshots on disk (read only). "event": the <NECIS_ID>.r folder of a per-event package, dated by the event's
# origin (a package shows the epoch in force then: GOCB.HHZ shows its 2021 epoch in the 2023-05 packages and its
# 2023-07-03 epoch in the 2024 and 2026 ones). "station": a station RESP fetch, dated as given.
SNAPSHOTS = [
    ("event", "/home/msseo/works/27.Haenam_sequence_renewal/data/stations/responses/necis_event_resp_20261014/*/RESP/*.r", None),   # GWANGYANG PORT: read from the Haenam tree
    ("event", "/home/msseo/works/27.Haenam_sequence_renewal/data/stations/responses/necis_event_resp_20261014/*/*/RESP/*.r", None),
    ("event", "/home/msseo/works/Claude/data/necis/events/*/RESP/*.r", None),          # the downloader's layout
    ("event", "/home/msseo/works/Claude/data/necis/events/*/*/RESP/*.r", None),
    ("event", "/home/msseo/works/15.PocketQuake/external/korea-cluster-relocation/*/kma_waveforms/RESP/*.r", None),
    ("event", "/home/msseo/works/10.Earthquake_cycle_project/*/kma_waveforms/RESP/*.r", None),
    ("event", "/home/msseo/works/24.Slip_inversion/35.240612_Buan_EQ/resp_files/*.r", None),
    ("station", "/home/msseo/works/5.P_wave_moment_tensor_inversion/Response/NECIS/RESP_KS_*_20241223", "2024-12-23"),
    ("station", "/home/msseo/works/27.Haenam_sequence/data/stations/responses/fetched/extracted", "2026-06-14"),     # GWANGYANG PORT: Buan fetch, read from the Haenam tree
]
SENS_FREQS = (1.0, 5.0)        # Hz: analog sensitivity compared at both
SENS_TOL = 0.01                # 1 %
START_TOL_DAYS = 10            # a snapshot epoch and a merged epoch whose starts differ by <= this are "the same epoch"
OPEN_AFTER = "2099-01-01"      # an end date on or after this is an open epoch
USE_SNAPSHOTS = True           # P1, DECIDED by the user 2026-10-14 (rule M4, and snapshots in M5): True uses NECIS snapshot epochs where no
                               # primary epoch is confirmed. False (the conservative default) uses the four primary
                               # sources only; snapshots then serve only as evidence (confirmation, conflicts, #20).
                               # coverage.csv gives the other setting's numbers in its *_alt columns.
# X (conservative default): spans where the waveform checks (waveform_checks.py) show that NO documented response in
# force fits the data; the merged inventory carries no response there, so nothing downstream uses a known-wrong gain.
# Each line names the proposal (README.md) that would replace the exclusion. Remove a line to restore the merged epoch.
# The user's decisions of 2026-10-14 (README.md "Decisions"): P1 yes except JDO (SNAPSHOT_SKIP), P2 yes
# (MIDNIGHT_END_INCLUSIVE), P3 = NECIS's GS-13 response for JDO2.EL 2018-12-19..2026-01-27 (OVERRIDE), P4 KWJU and P5 SEJF
# stay excluded (left out of the magnitudes), WAN2 keeps the master.
SNAPSHOT_SKIP = [("KS", "JDO")]   # JDO (2018): the only snapshot epoch is NECIS's stale one from 2000-12-17; its 2 gap days stay open
MIDNIGHT_END_INCLUSIVE = True     # P2: an epoch ending at 00:00:00 of day X also covers day X when no other epoch covers that day
OVERRIDE = [
    dict(net="KS", sta="JDO2", band="EL", start="2018-12-19", end="2026-01-28", src="NECIS event RESP", rec_start="2022-01-27",
         why="P3 (user, 2026-10-14): NECIS's GS-13 response (gain 18.3x the master's) for the whole span: the raw count level "
             "is constant 2019-03 .. 2026-01-27 and drops 22x on 2026-01-28 with NECIS's gain change (waveform_checks T5)"),
]
EXCLUDE = [
    dict(net="KS", sta="KWJU", band="HH", start="2015-10-01", end=None,
         why="neither source fits the co-located KWJ2: KWJU/KWJ2 velocity ratio 0.28 with the master, 0.16 with NECIS, "
             "0.7-3 Hz, 9 events 2019-2024 (waveform_checks T3) (P6)"),
    dict(net="KS", sta="SEJF", band="EL", start="2024-12-22", end=None,
         why="its only response (NECIS snapshot, CMG-40T-1-B T43959) gives 0.82-0.87 of the co-located SEJF.HG; the "
             "master response of the same sensor serial at YOA gives 0.99-1.01 (waveform_checks T2) (P7)"),
]
LOC = {"KS": "", "KG": "", "JU": "00"}        # location codes of the data (checked on day files of each network)
DEFAULT_RATE = {"KS": 100.0, "KG": 100.0, "JU": 200.0}
# remove_response with the merged XML on 2 h of one day: one station per network, a NECIS-current-only station and a
# snapshot-only station
VALIDATE = [("KS", "GWYB", "HHZ", "2021-08-28"), ("KG", "JRB", "HGZ", "2021-08-28"), ("KS", "BLLA", "HGZ", "2026-01-01"),   # GWANGYANG PORT
            ("KS", "BAU", "ELZ", "2021-08-28"), ("KS", "PHDA", "HGZ", "2026-01-01")]

PRIMARY = ("master XML", "NECIS current", "KIGAM", "JU StationXML")
FAMILY_RANK = {"master XML": 0, "NECIS current": 1, "KIGAM": 2, "JU StationXML": 3, "NECIS event RESP": 4,
               "NECIS station RESP": 5}
INF = float("inf")


def ts(x):
    from obspy import UTCDateTime
    return UTCDateTime(str(x)).timestamp


def day_ts(o):
    """POSIX time of 00:00 UTC of the day with ordinal o"""
    return (o - 719163) * 86400.0


def next_midnight(t):
    return (np.floor(t / 86400.0) + 1) * 86400.0


def iso(t):
    from obspy import UTCDateTime
    return "open" if t == INF else str(UTCDateTime(t))[:19]


def day_of(t):
    from obspy import UTCDateTime
    return "open" if t == INF else str(UTCDateTime(t))[:10]


@dataclass(eq=False)
class Rec:
    src: str                  # family (FAMILY_RANK)
    inst: str                 # instance label
    net: str
    sta: str
    loc: str
    cha: str
    t0: float
    t1: float                 # INF for an open epoch
    start_raw: str
    end_raw: str
    issued: float             # end of the source's issue day: the source states the state up to here
    channel: object = field(repr=False)
    desc: str = ""
    sens_value: float = np.nan
    sens_freq: float = np.nan
    amp: tuple = (np.nan, np.nan)     # analog sensitivity |H_a| at SENS_FREQS
    full: tuple = (np.nan, np.nan)    # full-cascade |H| at SENS_FREQS (with the FIR stages)
    sign: int = 0                     # sign of the product of the stage gains (information only)
    phase: float = np.nan             # phase of the analog response at SENS_FREQS[-1] (polarity test)
    suspect: str = ""
    path: str = ""
    insts: list = field(default_factory=list)
    cont: float = -INF        # master open epoch: a later source shows the same epoch still running up to here

    @property
    def own_conf(self):
        return min(self.t1, self.issued)

    @property
    def conf_until(self):
        return max(min(self.t1, self.issued), min(self.t1, self.cont))


# ------------------------------------------------------------------------------------------------------------ readers
def chosen_streams():
    st = pd.read_csv(STATION_TABLE)
    out = [dict(net=r.Network, sta=r.Code, band=r.Sensor, loc=LOC[r.Network], lat=r.Latitude, lon=r.Longitude,
                elev=r.Elevation) for r in st.itertuples()]
    reg = pd.read_csv(REGISTRY_STREAMS)
    regd = {(r.net, r.sta): r.sensor for r in reg.itertuples()}
    tab = {(o["net"], o["sta"]): o["band"] for o in out}
    diff = [k for k in regd if tab.get(k) != regd[k]]
    only_tab = sorted(f"{a}.{b}" for a, b in tab if (a, b) not in regd)
    print(f"chosen streams: {len(out)} codes in the station table {pd.Series([o['net'] for o in out]).value_counts().to_dict()}; "
          f"registry/streams.csv has {len(regd)} codes, chosen sensor differs at {len(diff)} {diff}; "
          f"only in the station table (2018-2019 predecessor codes): {only_tab}")
    return out


def data_days(streams):
    """(net, sta, cha) -> sorted day ordinals with a day file in the link tree within W0..W1"""
    d0, d1 = date.fromisoformat(W0).toordinal(), date.fromisoformat(W1).toordinal()
    out = {}
    for s in streams:
        for comp in "ZNE":
            cha = s["band"] + comp
            d = LINK_TREE / s["net"] / s["sta"] / f"{cha}.D"
            days = []
            if d.is_dir():
                for f in os.listdir(d):
                    m = re.search(r"\.(\d{4})\.(\d{3})$", f)
                    if m:
                        o = (date(int(m.group(1)), 1, 1) + timedelta(days=int(m.group(2)) - 1)).toordinal()
                        if d0 <= o <= d1:
                            days.append(o)
            out[(s["net"], s["sta"], cha)] = sorted(days)
    return out


def responses_of(ch):
    """(analog |H_a|, full |H|, polarity, suspect) at SENS_FREQS"""
    from obspy.core.inventory.response import PolesZerosResponseStage
    r = ch.response
    f = np.array(SENS_FREQS)
    full = (np.nan,) * len(f)
    try:
        full = tuple(float(x) for x in abs(r.get_evalresp_response_for_frequencies(
            f, output="DEF", hide_sensitivity_mismatch_warning=True)))
    except Exception as e:                                            # noqa: BLE001 - logged, kept as NaN
        print(f"  evalresp failed for {ch.code} {ch.start_date}: {e}")
    hc = np.ones(len(f), dtype=complex)          # analog response: poles-zeros stages evaluated, other stages' signed gain
    suspect = []
    for k, s in enumerate(r.response_stages, start=1):
        if isinstance(s, PolesZerosResponseStage):
            try:
                hc *= r.get_evalresp_response_for_frequencies(f, output="DEF", start_stage=k, end_stage=k,
                                                              hide_sensitivity_mismatch_warning=True)
            except Exception as e:                                    # noqa: BLE001
                print(f"  evalresp (stage {k}) failed for {ch.code} {ch.start_date}: {e}")
                hc *= np.nan
            if k == 1 and len(s.poles) == 0:
                suspect.append("sensor stage has no poles (flat response)")
        elif s.stage_gain not in (None, 0):
            hc *= s.stage_gain
    g = [s.stage_gain for s in r.response_stages if s.stage_gain not in (None, 0)]
    sign = int(np.sign(np.prod(g))) if g else 0
    return tuple(float(x) for x in abs(hc)), full, sign, "; ".join(suspect), float(np.angle(hc[-1]))


def equip(ch, path=None):
    """sensor / datalogger: the RESP file's first comment line, else the StationXML equipment fields"""
    if path and os.path.basename(path).startswith("RESP"):
        try:
            first = open(path, errors="replace").readline().strip()
            if first.startswith("#") and first.lstrip("#").strip():
                return first.lstrip("#").strip()
        except OSError:
            pass
    parts = []
    for e in (ch.sensor, ch.data_logger):
        if e is not None:
            t = " ".join(str(x) for x in (e.manufacturer, e.model, e.description) if x not in (None, ""))
            if t:
                parts.append(t)
    return " / ".join(parts)


def make_rec(src, inst, net, ch, issued_day, path=""):
    end = ch.end_date
    t1 = INF if (end is None or str(end) >= OPEN_AFTER) else end.timestamp
    s = ch.response.instrument_sensitivity if ch.response else None
    amp, full, sign, suspect, phase = responses_of(ch)
    return Rec(src=src, inst=inst, net=net, sta=ch._sta, loc=ch.location_code, cha=ch.code, t0=ch.start_date.timestamp,
               t1=t1, start_raw=str(ch.start_date)[:19], end_raw=str(end)[:19] if end else "None",
               issued=next_midnight(issued_day), channel=ch, desc=equip(ch, path),
               sens_value=s.value if s else np.nan, sens_freq=s.frequency if s else np.nan, amp=amp, full=full,
               sign=sign, suspect=suspect, path=path, phase=phase)


def channels_of(inv, net, sta, band):
    out = []
    for n in inv.select(network=net, station=sta, channel=band + "?"):
        for s in n:
            for c in s:
                if c.code[-1] in "ZNE":
                    c._sta = s.code
                    out.append(c)
    return out


def read_primary(streams):
    from obspy import read_inventory
    recs = []
    m = read_inventory(str(MASTER_XML))
    print(f"master XML: {MASTER_XML.name}, created {str(m.created)[:19]}")
    kf = sorted(glob.glob(KIGAM_XML_GLOB))[-1]
    k = read_inventory(kf)
    print(f"KIGAM: {os.path.basename(kf)}, created {str(k.created)[:19]}")
    j = read_inventory(str(JU_XML))
    print(f"JU StationXML: {JU_XML}, created {str(j.created)[:19]}")
    necis = {}
    for f in sorted(glob.glob(NECIS_CURRENT_GLOB)):
        mm = re.search(r"RESP\.([A-Z0-9]+)\.([A-Z0-9]+)\.([A-Z0-9]*)\.([A-Z0-9]{3})$", f)
        if mm:
            necis[(mm.group(1), mm.group(2), mm.group(4))] = f
    for s in streams:
        net, sta, band = s["net"], s["sta"], s["band"]
        for c in channels_of(m, net, sta, band):
            recs.append(make_rec("master XML", MASTER_XML.name, net, c, m.created.timestamp, str(MASTER_XML)))
        if net == "KG":
            for c in channels_of(k, net, sta, band):
                recs.append(make_rec("KIGAM", os.path.basename(kf), net, c, k.created.timestamp, kf))
        if net == "JU":
            for c in channels_of(j, net, sta, band):
                recs.append(make_rec("JU StationXML", JU_XML.name, net, c, j.created.timestamp, str(JU_XML)))
        for comp in "ZNE":
            f = necis.get((net, sta, band + comp))
            if f:
                for c in channels_of(read_inventory(f), net, sta, band):
                    recs.append(make_rec("NECIS current", f.split("/responses/")[-1], net, c, ts(NECIS_CURRENT_ISSUED), f))
    return recs


def snapshot_dirs():
    """[(kind, folder, label, date)] for every NECIS RESP snapshot folder found"""
    cat = pd.read_csv(NECIS_CATALOG, encoding="utf-8-sig", dtype={"necis_id": str}).set_index("necis_id")
    out, seen = [], set()
    for kind, pat, when in SNAPSHOTS:
        for d in sorted(glob.glob(pat)):
            real = os.path.realpath(d)
            if real in seen or not os.path.isdir(d):
                continue
            seen.add(real)
            if kind == "event":
                eid = os.path.basename(d).split(".")[0]
                if eid not in cat.index:
                    print(f"  snapshot {d}: NECIS id {eid} is not in the NECIS catalog; skipped")
                    continue
                out.append((kind, d, f"event {eid}", str(cat.loc[eid, "origin_utc"])))
            else:
                out.append((kind, d, f"station fetch {os.path.basename(d)}", when))
    return out


def read_snapshots(streams, snaps):
    """one Rec per snapshot RESP file of a chosen channel"""
    from obspy import read_inventory
    want = {(s["net"], s["sta"]): s["band"] for s in streams if s["net"] != "JU"}
    recs = []
    for kind, d, label, when in snaps:
        for (net, sta), band in want.items():
            for comp in "ZNE":
                f = os.path.join(d, f"RESP.{net}.{sta}..{band}{comp}")
                if os.path.exists(f):
                    for c in channels_of(read_inventory(f), net, sta, band):
                        r = make_rec("NECIS event RESP" if kind == "event" else "NECIS station RESP",
                                     f"{label} ({when[:10]})", net, c, ts(when), f)
                        r.insts = [f"{label} ({when[:10]})"]
                        r.snap_day = when[:10]
                        r.snap_time = ts(when) + (43200.0 if len(when) <= 10 else 0.0)   # a bare date: its noon
                        recs.append(r)
    return recs


def dedup_snapshots(snap):
    """one Rec per distinct snapshot epoch (channel, start, analog sensitivity, polarity); its issue day is the latest
    snapshot that shows it"""
    groups = {}
    for r in snap:
        key = (r.net, r.sta, r.cha, r.start_raw, tuple(float(f"{x:.5g}") for x in r.amp), r.sign)
        if key not in groups:
            g = copy.copy(r)
            g.insts = list(r.insts)
            groups[key] = g
        else:
            g = groups[key]
            g.insts += r.insts
            if r.issued > g.issued:
                g.issued, g.path = r.issued, r.path
    for g in groups.values():
        g.inst = f"{len(g.insts)} snapshot(s), {g.insts[0].split(' (')[1][:-1]}..{day_of(g.issued - 1)}"
    return list(groups.values())


# -------------------------------------------------------------------------------------------------------------- merge
@dataclass(eq=False)
class Piece:
    t0: float
    t1: float
    rec: Rec
    role: str


def choose(cands, latest_master, use_snap):
    """the merged record at one instant; cands = [(rec, confirmed)]"""
    def best(rs, key):
        return max(rs, key=key)
    conf_p = [r for r, c in cands if c and r.src in PRIMARY]
    if conf_p:
        newer = [r for r in conf_p if r.src in ("NECIS current", "KIGAM") and r.t0 > latest_master + START_TOL_DAYS * 86400.0]
        if newer:
            return best(newer, lambda r: (r.t0, -FAMILY_RANK[r.src])), ("M2 newer than the master" if latest_master > -INF
                                                                         else "M2 only primary source")
        masters = [r for r in conf_p if r.src == "master XML"]
        if masters:
            return best(masters, lambda r: r.t0), "M1 master history"
        r = best(conf_p, lambda r: (r.t0, -FAMILY_RANK[r.src]))
        return r, ("M3 JU (only source)" if r.src == "JU StationXML" else "M3 NECIS/KIGAM where no master epoch is confirmed")
    if use_snap:
        conf_s = [r for r, c in cands if c and r.src not in PRIMARY]
        if conf_s:
            return best(conf_s, lambda r: (r.t0, r.issued)), "M4 snapshot (PROPOSAL)"
    pool = [r for r, c in cands if r.src in PRIMARY or use_snap]
    if pool:
        r = best(pool, lambda r: (r.issued, r.t0, -FAMILY_RANK[r.src]))
        same = [q for q in pool if q.src == "master XML" and r.src != "master XML"
                and r.t0 <= q.t0 + START_TOL_DAYS * 86400.0]
        if same:                     # the latest statement describes the master's epoch: keep the master's description
            return best(same, lambda q: q.t0), "M5 extrapolated"
        return r, "M5 extrapolated" + (" snapshot (PROPOSAL)" if r.src not in PRIMARY else "")
    return None, ""


def excluded_spans(key):
    out = []
    for x in EXCLUDE:
        if (x["net"], x["sta"]) == key[:2] and key[2].startswith(x["band"]):
            out.append((ts(x["start"]), ts(x["end"]) if x["end"] else INF, x["why"]))
    return out


def merge_channel(recs, use_snap, key=None):
    """merged pieces of one channel: the rule of choose() on every elementary interval between epoch boundaries"""
    all_recs = recs
    skip = key is not None and key[:2] in SNAPSHOT_SKIP
    recs = [r for r in recs if r.src in PRIMARY or (use_snap and not skip)]
    if not recs:
        return []
    latest_master = max((r.t0 for r in recs if r.src == "master XML"), default=-INF)
    b = sorted({x for r in recs for x in (r.t0, r.t1, r.conf_until) if np.isfinite(x)})
    edges = b + [INF]
    pieces = []
    for a, z in zip(edges[:-1], edges[1:]):
        m = a + 1.0 if z == INF else (a + z) / 2
        cov = [r for r in recs if r.t0 <= m < r.t1]
        cands = [(r, m < r.conf_until) for r in cov if not superseded(r, recs, m, m + 1e-3)]
        r, role = choose(cands, latest_master, use_snap)
        if r is None:
            continue
        if pieces and pieces[-1].rec is r and pieces[-1].role == role and abs(pieces[-1].t1 - a) < 1e-6:
            pieces[-1].t1 = z
        else:
            pieces.append(Piece(a, z, r, role))
    # an open piece ends where its record ends
    for p in pieces:
        p.t1 = min(p.t1, p.rec.t1)
    # the same record may continue across a confirmed -> extrapolated boundary: one epoch in the XML
    out = []
    for p in pieces:
        if out and out[-1].rec is p.rec and abs(out[-1].t1 - p.t0) < 1e-6:
            out[-1].t1 = p.t1
            if p.role != out[-1].role and p.role not in out[-1].role and p.t0 <= ts(W1) + 86400:
                out[-1].role = f"{out[-1].role}, then {p.role} from {day_of(p.t0)}"
        else:
            out.append(p)
    # keep the epochs that touch the window; the one running at its end keeps its own end (open if open)
    w0, w1 = ts(W0), ts(W1) + 86400.0
    out = [p for p in out if p.t1 > w0 and p.t0 < w1]
    if out and out[-1].t1 >= w1:
        out[-1].t1 = out[-1].rec.t1
    # P3: an override replaces the merged response over its span by the named source epoch
    for o in [o for o in OVERRIDE if key and (o["net"], o["sta"]) == key[:2] and key[2].startswith(o["band"])]:
        a, b = ts(o["start"]), ts(o["end"]) if o["end"] else INF
        cand = [r for r in all_recs if r.src == o["src"] and abs(r.t0 - ts(o["rec_start"])) < 86400.0]
        assert cand, f"override {o['sta']}: no {o['src']} epoch starting {o['rec_start']} on {key}"
        r = max(cand, key=lambda r: r.issued)
        cut = []
        for p in out:
            if p.t1 <= a or p.t0 >= b:
                cut.append(p)
                continue
            if p.t0 < a:
                cut.append(Piece(p.t0, a, p.rec, p.role))
            if p.t1 > b:
                cut.append(Piece(b, p.t1, p.rec, p.role))
        cut.append(Piece(a, b, r, "P3 override (user 2026-10-14)"))
        out = sorted(cut, key=lambda p: p.t0)
    # P2: an end at 00:00:00 of day X also covers day X where no other piece does
    if MIDNIGHT_END_INCLUSIVE:
        out = sorted(out, key=lambda p: p.t0)
        for i, p in enumerate(out):
            if np.isfinite(p.t1) and abs(p.t1 / 86400.0 - round(p.t1 / 86400.0)) < 1e-9:
                nxt = out[i + 1].t0 if i + 1 < len(out) else INF
                if nxt > p.t1 + 1.0:
                    p.t1 = min(p.t1 + 86400.0, nxt)
                    p.role = f"{p.role}; P2 end read as through that day"
    # X: cut the excluded spans out
    for a, b, _ in excluded_spans(key) if key else []:
        cut = []
        for p in out:
            if p.t1 <= a or p.t0 >= b:
                cut.append(p)
                continue
            if p.t0 < a:
                cut.append(Piece(p.t0, a, p.rec, p.role))
            if p.t1 > b:
                cut.append(Piece(b, p.t1, p.rec, p.role))
        out = cut
    return out


# ----------------------------------------------------------------------------------------------------------- per day
def agree(a, b):
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return False
    return all(abs(x / y - 1) <= SENS_TOL for x, y in zip(a, b))


def differ(a, b):
    """two statements give different responses: |H_a| disagrees beyond SENS_TOL, or the polarity differs"""
    return (not agree(a.amp, b.amp)) or opposite(a, b)


def opposite(a, b):
    """opposite polarity: the analog responses at SENS_FREQS[-1] are more than 90 degrees apart in phase (this
    includes a negative stage gain, a negative normalization factor or a negative sensitivity, wherever it sits)"""
    return bool(np.isfinite(a.phase) and np.isfinite(b.phase) and np.cos(a.phase - b.phase) < 0)


def superseded(r, others, a, b):
    """True when NECIS itself later restated the time [a, b) differently: a NECIS-family record issued later than r,
    confirmed over that time, with a different response (NECIS revising its own database, e.g. a corrected Centaur gain
    or an epoch entered with a back-dated start). Only NECIS statements supersede NECIS statements."""
    if not r.src.startswith("NECIS"):
        return False
    return any(q is not r and q.src.startswith("NECIS") and q.issued > r.issued and q.t0 < b and a < q.conf_until
               and differ(q, r) for q in others)


def maxdiff(a, b):
    return max(abs(x / y - 1) for x, y in zip(a, b)) if np.all(np.isfinite(a)) and np.all(np.isfinite(b)) else np.nan


def evaluate_channel(key, pieces, recs, days):
    excl = excluded_spans(key)
    union = []
    for p in sorted(pieces, key=lambda p: p.t0):
        if union and p.t0 <= union[-1][1] + 1.5:
            union[-1][1] = max(union[-1][1], p.t1)
        else:
            union.append([p.t0, p.t1])
    out = []
    for o in days:
        t0 = day_ts(o)
        t1 = t0 + 86400.0
        over = [p for p in pieces if p.t0 < t1 and p.t1 > t0]
        if any(a <= t0 + 0.5 and b >= t1 - 1.5 for a, b in union):
            status = "covered"
        elif any(a < t1 and b > t0 for a, b, _ in excl):
            status = "excluded"
        elif over:
            status = "partial"
        else:
            hole = any(abs(p.t1 - t0) < 1.5 for p in pieces) and any(abs(p.t0 - t1) < 1.5 for p in pieces)
            status = "boundary hole" if hole else "none"
        merged = None
        if over:
            noon = [p for p in over if p.t0 <= t0 + 43200 < p.t1]
            merged = (noon or sorted(over, key=lambda p: -(min(p.t1, t1) - max(p.t0, t0))))[0]
        stat = [r for r in recs if r.t0 < t1 and r.t1 > t0 and t0 < r.conf_until]    # confirmed statements that day
        revised = [r for r in stat if superseded(r, recs, t0, t1)]
        stat = [r for r in stat if all(r is not x for x in revised)]
        confirmed_by = ""
        pool = list(stat)
        if merged is not None:
            mr = merged.rec
            if t0 < mr.own_conf:
                confirmed_by = mr.src
            elif t0 < mr.conf_until:
                confirmed_by = f"{mr.src} (epoch continuity)"
            else:
                ok = sorted({r.src for r in stat if agree(r.amp, mr.amp)})
                confirmed_by = ok[0] if ok else ""
            if all(mr is not r for r in stat):
                pool.append(mr)
        bad, sgn = set(), set()
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                a, b = pool[i], pool[j]
                if a.src == b.src and a.inst == b.inst:
                    continue
                d = maxdiff(a.amp, b.amp)
                if np.isfinite(d) and d > SENS_TOL:
                    bad.add(f"{a.src} {day_of(a.t0)} vs {b.src} {day_of(b.t0)} ({100 * d:.1f}%)")
                if opposite(a, b):
                    sgn.add(f"{a.src} {day_of(a.t0)} ({np.degrees(a.phase):+.0f} deg) vs {b.src} {day_of(b.t0)} "
                            f"({np.degrees(b.phase):+.0f} deg)")
        out.append(dict(net=key[0], sta=key[1], cha=key[2], ordinal=o, status=status,
                        source=merged.rec.src if merged else "", role=merged.role if merged else "",
                        suspect=merged.rec.suspect if merged else "", confirmed_by=confirmed_by,
                        conflict="; ".join(sorted(bad)), sign_conflict="; ".join(sorted(sgn)),
                        revised="; ".join(sorted({f"{r.src} {day_of(r.t0)} ({r.inst})" for r in revised}))))
    return out


def ranges(ordinals):
    if not len(ordinals):
        return ""
    o = sorted(ordinals)
    runs, a, b = [], o[0], o[0]
    for x in o[1:]:
        if x == b + 1:
            b = x
        else:
            runs.append((a, b))
            a = b = x
    runs.append((a, b))
    return "; ".join(f"{date.fromordinal(a)}" + (f"..{date.fromordinal(b)}" if b > a else "") + f" ({b - a + 1} d)"
                     for a, b in runs)


# ------------------------------------------------------------------------------------------------------------ outputs
def build_xml(streams, merged):
    from obspy import UTCDateTime
    from obspy.core.inventory import Comment, Inventory, Network, Station
    nets = {}
    for s in streams:
        net = nets.setdefault(s["net"], Network(code=s["net"], stations=[]))
        chans = []
        for comp in "ZNE":
            for p in merged.get((s["net"], s["sta"], s["band"] + comp), []):
                c = copy.deepcopy(p.rec.channel)
                if hasattr(c, "_sta"):
                    del c._sta
                c.location_code = s["loc"]
                c.start_date = UTCDateTime(p.t0)
                c.end_date = None if p.t1 == INF else UTCDateTime(p.t1)
                c.latitude, c.longitude, c.elevation = s["lat"], s["lon"], s["elev"]
                note = []
                if p.rec.src.startswith("NECIS") or c.depth in (None, 123456.0):
                    c.depth = 0.0
                    note.append("depth not given by the source (RESP): 0.0 is a placeholder")
                if c.azimuth is None or c.dip is None:
                    c.azimuth, c.dip = {"Z": (0.0, -90.0), "N": (0.0, 0.0), "E": (90.0, 0.0)}[c.code[-1]]
                    note.append("azimuth/dip nominal (not in the source)")
                if not c.sample_rate:
                    c.sample_rate = DEFAULT_RATE[s["net"]]
                if p.rec.suspect:
                    note.append(f"SUSPECT: {p.rec.suspect}")
                c.comments = [Comment(value=(f"source: {p.rec.src} ({p.rec.inst}); file {p.rec.path}; source epoch "
                                             f"{p.rec.start_raw}..{p.rec.end_raw}; rule: {p.role}"
                                             + ("; " + "; ".join(note) if note else "")
                                             + "; orientation: use registry/orientation.py, not this file"))]
                chans.append(c)
        st = Station(code=s["sta"], latitude=s["lat"], longitude=s["lon"], elevation=s["elev"], channels=chans,
                     start_date=min((c.start_date for c in chans), default=None))
        ex = [x for x in EXCLUDE if (x["net"], x["sta"]) == (s["net"], s["sta"])]
        st.comments = [Comment(value=f"EXCLUDED {x['band']} {x['start']}..{x['end'] or 'open'}: {x['why']}") for x in ex]
        if not chans:
            st.comments.append(Comment(value="no response for the chosen sensor in this inventory"))
        net.stations.append(st)
    return Inventory(networks=list(nets.values()), source="Haenam renewal R1d merged response inventory "
                     "(docs/decisions/response_inventory_20261014/build_response_inventory.py)")


def validate_remove_response(path, inv_alt=None):
    """remove_response with the merged XML on 2 h of one day; compared with dividing by the full |H(5 Hz)|. A channel
    with no epoch that day in the merged XML is tried in the other USE_SNAPSHOTS setting's inventory (inv_alt, in
    memory, not written), so a snapshot-derived response is evaluated too."""
    from obspy import UTCDateTime, read, read_inventory
    inv_main = read_inventory(str(path))
    rows = []
    for net, sta, cha, day in VALIDATE:
        dd = date.fromisoformat(day)
        f = LINK_TREE / net / sta / f"{cha}.D" / f"{net}.{sta}..{cha}.D.{dd.year}.{dd.timetuple().tm_yday:03d}"
        if not f.exists():
            rows.append(dict(id=f"{net}.{sta}.{cha}", day=day, result="no day file"))
            continue
        st = read(str(f))
        st.merge(method=1, fill_value=0)
        tr = st[0]
        tr.trim(UTCDateTime(day) + 3600, UTCDateTime(day) + 3 * 3600)
        inv, which = inv_main, "merged_inventory.xml"
        if not len(inv.select(network=net, station=sta, channel=cha, time=tr.stats.starttime).get_contents()["channels"]):
            inv, which = inv_alt, f"USE_SNAPSHOTS={not USE_SNAPSHOTS} alternative (not written)"
        try:
            comment = [c for c in inv.select(network=net, station=sta, location=tr.stats.location, channel=cha,
                                              time=tr.stats.starttime)[0][0][0].comments][0].value
            resp = inv.get_response(tr.id, tr.stats.starttime)
            h5 = abs(resp.get_evalresp_response_for_frequencies(np.array([5.0]), output="DEF",
                                                                hide_sensitivity_mismatch_warning=True)[0])
            out = "ACC" if cha[:2] == "HG" else "VEL"
            a = tr.copy()
            a.detrend("demean"); a.taper(0.05)
            a.remove_response(inventory=inv, output=out, pre_filt=(0.5, 1.0, 20.0, 30.0), water_level=None)
            b = tr.copy()
            b.detrend("demean"); b.taper(0.05)
            b.data = b.data / h5
            for x in (a, b):
                x.filter("bandpass", freqmin=3.0, freqmax=8.0, corners=4, zerophase=True)
            ra, rb = float(np.sqrt(np.mean(a.data ** 2))), float(np.sqrt(np.mean(b.data ** 2)))
            rows.append(dict(id=tr.id, day=day, inventory=which, units=out, source=comment.split(";")[0].replace("source: ", ""),
                             finite=bool(np.all(np.isfinite(a.data))), rms_3_8Hz=f"{ra:.3e}", ratio=round(ra / rb, 3),
                             result="OK" if np.all(np.isfinite(a.data)) and 0.8 < ra / rb < 1.25 else "CHECK"))
        except Exception as e:                                         # noqa: BLE001
            rows.append(dict(id=tr.id, day=day, result=f"FAILED: {e}"))
    return pd.DataFrame(rows)


def reader_crosscheck(recs, streams):
    """my Z rows of the four primary sources vs the station-inventory table (same files, same reader)"""
    old = pd.read_csv(OLD_EPOCH_TABLE)
    want = {(s["sta"], s["band"]) for s in streams}
    old = old[[(a, b) in want for a, b in zip(old.sta, old.sensor)]]
    mine = pd.DataFrame([dict(sta=r.sta, sensor=r.cha[:2], start=r.start_raw[:10], end=r.end_raw[:10],
                              sensitivity=r.sens_value, source=r.src) for r in recs if r.cha.endswith("Z")])
    k = ["sta", "sensor", "start", "source"]
    a, b = old.groupby(k).sensitivity.first(), mine.groupby(k).sensitivity.first()
    both = a.index.intersection(b.index)
    same = int(sum(np.isclose(a[i], b[i], rtol=1e-9) for i in both))
    extra = b.index.difference(a.index)
    pre = mine.set_index(k).loc[list(extra)] if len(extra) else mine.iloc[:0]
    print(f"reader cross-check (Z, chosen sensors) vs {OLD_EPOCH_TABLE.relative_to(ROOT)}: old {len(a)} rows, mine {len(b)}; "
          f"common {len(both)}, identical sensitivity {same}; only in the old table {len(a.index.difference(b.index))}; "
          f"only in mine {len(extra)} (all ending before 2020-01-01, the old table's window start: "
          f"{bool((pre.end < '2020-01-01').all()) if len(pre) else True})")


# --------------------------------------------------------------------------------------------------------------- main
def main():
    streams = chosen_streams()
    days = data_days(streams)
    print(f"data days: {sum(len(v) for v in days.values())} channel-days in {W0}..{W1} "
          f"(link tree {LINK_TREE.relative_to(ROOT)})")
    prim = read_primary(streams)
    reader_crosscheck(prim, streams)
    snaps = snapshot_dirs()
    print(f"NECIS RESP snapshots found: {len(snaps)}")
    for kind, d, label, when in snaps:
        print(f"   {label:36s} {when[:19]}  {d}")
    in_seq = [x for x in snaps if any(a <= x[3][:10] <= b for a, b in SEQ.values())]
    print(f"   inside the two sequences {SEQ}: {len(in_seq)}")
    snap_files = read_snapshots(streams, snaps)
    snap = dedup_snapshots(snap_files)
    print(f"snapshot RESP files of chosen channels: {len(snap_files)} -> {len(snap)} distinct snapshot epochs")
    allrecs = prim + snap
    by_ch = {}
    for r in allrecs:
        by_ch.setdefault((r.net, r.sta, r.cha), []).append(r)
    # epoch continuity (rule M1): a master epoch still open at the master's issue day stays confirmed as long as a later
    # source (NECIS current, KIGAM, a snapshot) shows an epoch that began no later than START_TOL_DAYS after it and is
    # still running, i.e. no instrument change since; the master's description of that epoch is kept (any disagreement
    # with the later description is a conflict, not a new epoch)
    n_cont = 0
    for recs in by_ch.values():
        for r in recs:
            if r.src == "master XML" and r.t1 == INF:
                later = [q.own_conf for q in recs if q.src != "master XML" and q.t0 <= r.t0 + START_TOL_DAYS * 86400.0
                         and q.own_conf > r.own_conf]
                if later:
                    r.cont = max(later)
                    n_cont += 1
    print(f"master open epochs confirmed past the master's issue day by epoch continuity: {n_cont}")

    merged, merged_p, merged_p20, rows_d = {}, {}, {}, []
    for s in streams:
        for comp in "ZNE":
            key = (s["net"], s["sta"], s["band"] + comp)
            recs = by_ch.get(key, [])
            merged[key] = merge_channel(recs, USE_SNAPSHOTS, key)
            merged_p[key] = merge_channel(recs, not USE_SNAPSHOTS, key) # the other setting of USE_SNAPSHOTS (*_alt)
            merged_p20[key] = merge_channel(recs, False)                # ... without the exclusions X (ISSUES #20 check)
            ev = evaluate_channel(key, merged[key], recs, days[key])
            evp = evaluate_channel(key, merged_p[key], recs, days[key])
            for x, y in zip(ev, evp):
                x["status_alt"] = y["status"]
                x["confirmed_alt"] = y["confirmed_by"]
            rows_d += ev
    D = pd.DataFrame(rows_d)

    # ---- merged_inventory.xml
    inv = build_xml(streams, merged)
    xml = HERE / "merged_inventory.xml"
    try:
        inv.write(str(xml), format="STATIONXML", validate=True)
        print(f"wrote {xml.name} (validated against the FDSN StationXML schema)")
    except Exception as e:                                             # noqa: BLE001
        print(f"schema validation not possible ({e}); written without it")
        inv.write(str(xml), format="STATIONXML")
    from obspy import read_inventory
    back = read_inventory(str(xml))
    print(f"read back: {len(back.get_contents()['channels'])} channel epochs in {len(back.get_contents()['stations'])} stations")

    # ---- epochs_chosen.csv, source_epochs.csv
    rows = []
    for (net, sta, cha), pieces in merged.items():
        dset = days[(net, sta, cha)]
        for p in pieces:
            r = p.rec
            n_in = sum(1 for o in dset if p.t0 < day_ts(o) + 86400 and p.t1 > day_ts(o))
            rows.append(dict(net=net, sta=sta, loc=LOC[net], cha=cha, start=iso(p.t0), end=iso(p.t1), source=r.src,
                             source_instance=r.inst, rule=p.role, source_start=r.start_raw, source_end=r.end_raw,
                             confirmed_until=day_of(r.conf_until - 1) if np.isfinite(r.conf_until) else "open",
                             stated_sensitivity=r.sens_value, stated_frequency_hz=r.sens_freq,
                             analog_sens_1hz=r.amp[0], analog_sens_5hz=r.amp[1], full_resp_1hz=r.full[0],
                             full_resp_5hz=r.full[1], stage_gain_sign=r.sign,
                             analog_phase_deg=round(float(np.degrees(r.phase)), 1), suspect=r.suspect,
                             sensor_datalogger=r.desc, data_days_in_epoch=n_in, file=r.path))
    pd.DataFrame(rows).to_csv(HERE / "epochs_chosen.csv", index=False)
    used = {id(p.rec) for ps in merged.values() for p in ps}
    S = pd.DataFrame([dict(net=r.net, sta=r.sta, loc=r.loc, cha=r.cha, source=r.src, instance=r.inst, start=r.start_raw,
                           end=r.end_raw, confirmed_until=day_of(r.conf_until - 1) if np.isfinite(r.conf_until) else "open",
                           stated_sensitivity=r.sens_value, stated_frequency_hz=r.sens_freq, analog_sens_1hz=r.amp[0],
                           analog_sens_5hz=r.amp[1], full_resp_1hz=r.full[0], full_resp_5hz=r.full[1], stage_gain_sign=r.sign,
                           analog_phase_deg=round(float(np.degrees(r.phase)), 1), suspect=r.suspect, sensor_datalogger=r.desc, used_in_merge=id(r) in used,
                           snapshots="; ".join(r.insts), file=r.path) for r in allrecs])
    S.sort_values(["net", "sta", "cha", "start", "source"]).to_csv(HERE / "source_epochs.csv", index=False)

    # ---- coverage.csv: one row per stream; a day counts when ANY component with data that day has the property
    cov = []
    main0 = date.fromisoformat(MAIN0).toordinal()
    for s in streams:
        g = D[(D.net == s["net"]) & (D.sta == s["sta"])]

        def ds(mask):
            return sorted(set(g[mask].ordinal))
        alld = ds(g.status.notna())
        gap = ds(~g.status.isin(["covered", "excluded"]))
        gap_p = ds(~g.status_alt.isin(["covered", "excluded"]))
        excl = ds(g.status == "excluded")
        excl_why = "; ".join(x["why"] for x in EXCLUDE if (x["net"], x["sta"]) == (s["net"], s["sta"]))
        hole = ds(g.status == "boundary hole")
        snapd = ds((g.source != "") & ~g.source.isin(PRIMARY))
        unconf = ds((g.status == "covered") & (g.confirmed_by == ""))
        unconf_p = ds((g.status_alt == "covered") & (g.confirmed_alt == ""))
        conf = ds(g.conflict != "")
        sconf = ds(g.sign_conflict != "")
        susp = ds(g.suspect != "")
        revd = ds(g.revised != "")
        rev_txt = "; ".join(sorted(set("; ".join(g[g.revised != ""].revised).split("; ")) - {""}))
        conf_txt = "; ".join(sorted(set("; ".join(g[g.conflict != ""].conflict).split("; ")) - {""}))
        sconf_txt = "; ".join(sorted(set("; ".join(g[g.sign_conflict != ""].sign_conflict).split("; ")) - {""}))
        by_rule = g[g.status == "covered"].groupby("source").ordinal.nunique().to_dict()
        cov.append(dict(net=s["net"], sta=s["sta"], loc=s["loc"], band=s["band"],
                        first_data_day=str(date.fromordinal(alld[0])) if alld else "",
                        last_data_day=str(date.fromordinal(alld[-1])) if alld else "",
                        data_days=len(alld), data_days_main=sum(1 for o in alld if o >= main0),
                        covered_days=len(alld) - len(gap) - len(excl), days_without_response=len(gap),
                        days_without_response_main=sum(1 for o in gap if o >= main0), gap_ranges=ranges(gap),
                        boundary_hole_days=len(hole),
                        days_excluded=len(excl), days_excluded_main=sum(1 for o in excl if o >= main0),
                        excluded_ranges=ranges(excl), exclusion_reason=excl_why,
                        gap_days_by_component=" ".join(f"{c[-1]}:{len(set(g[(g.cha == c) & (g.status != 'covered')].ordinal))}"
                                                       for c in sorted(set(g.cha))),
                        days_by_source="; ".join(f"{k} {v}" for k, v in sorted(by_rule.items())),
                        days_on_snapshot_epochs=len(snapd),
                        alt_setting=f"USE_SNAPSHOTS={not USE_SNAPSHOTS}",
                        days_without_response_alt=len(gap_p), gap_ranges_alt=ranges(gap_p),
                        unconfirmed_days=len(unconf), unconfirmed_ranges=ranges(unconf),
                        unconfirmed_days_alt=len(unconf_p),
                        suspect_days=len(susp), suspect_ranges=ranges(susp),
                        conflict_days=len(conf), conflict_ranges=ranges(conf), conflicts=conf_txt,
                        sign_conflict_days=len(sconf), sign_conflict_ranges=ranges(sconf), sign_conflicts=sconf_txt,
                        necis_revised_days=len(revd), necis_revised_statements=rev_txt,
                        **{f"seq{k}_{name}": sum(1 for o in v if date.fromisoformat(a0).toordinal() <= o <= date.fromisoformat(a1).toordinal())
                           for k, (a0, a1) in SEQ.items()
                           for name, v in (("data_days", alld), ("without_response", gap), ("excluded", excl),
                                           ("unconfirmed", unconf), ("conflict", conf), ("sign_conflict", sconf))},
                        status=("NO DATA" if not alld else "FULLY COVERED" if not gap and not excl else
                                "EXCLUDED (X)" if not gap else "GAPS" if not excl else "GAPS + EXCLUDED (X)")))
    C = pd.DataFrame(cov)
    C.to_csv(HERE / "coverage.csv", index=False)

    # ---- issue20_necis_resp.csv: every snapshot RESP file vs the primary sources and the merged inventory that day
    rows = []
    for r in snap_files:
        # the snapshot day; for a snapshot epoch already closed by then, the last day of that epoch
        when = r.snap_time if r.snap_time < r.t1 else r.t1 - 43200.0
        o = date.fromisoformat(day_of(when)).toordinal()
        has_data = o in set(days.get((r.net, r.sta, r.cha), []))
        seq = next((k for k, (a, b) in SEQ.items() if a <= r.snap_day <= b), "")
        P = [p for p in merged_p20.get((r.net, r.sta, r.cha), []) if p.t0 <= when < p.t1]
        F = [p for p in merged.get((r.net, r.sta, r.cha), []) if p.t0 <= when < p.t1]
        if not P:
            verdict = "NOT IN THE PRIMARY SOURCES: no primary epoch on the snapshot day"
        else:
            p = P[0]
            dstart = (r.t0 - p.t0) / 86400.0
            if not agree(r.amp, p.rec.amp):
                d = f"{100 * maxdiff(r.amp, p.rec.amp):.1f}%"
                if dstart > START_TOL_DAYS:
                    verdict = f"EPOCH MISSING FROM THE PRIMARY SOURCES: NECIS starts a new epoch {dstart:.0f} d after the primary epoch, response differs {d}"
                elif dstart >= -START_TOL_DAYS:
                    verdict = f"SAME EPOCH, DIFFERENT RESPONSE ({d})"
                else:
                    verdict = f"DIFFERENT RESPONSE ({d}); NECIS's epoch starts {-dstart:.0f} d before the primary epoch (NECIS keeps one epoch per channel)"
                verdict += (" [primary epoch extrapolated that day]" if when >= p.rec.conf_until else " [primary epoch confirmed that day]")
            elif abs(dstart) <= START_TOL_DAYS:
                verdict = "same epoch"
            elif dstart > 0:
                verdict = f"same response; NECIS starts this epoch {dstart:.0f} d after the primary epoch's start"
            else:
                verdict = f"same response; NECIS starts this epoch {-dstart:.0f} d before the primary epoch's start"
            if opposite(r, p.rec):
                verdict += "; POLARITY differs"
        if superseded(r, by_ch.get((r.net, r.sta, r.cha), []), when, when + 1.0):
            verdict += "; NECIS later restated this day differently (revised)"
        if r.suspect:
            verdict += f"; snapshot response SUSPECT ({r.suspect})"
        p, f = (P[0] if P else None), (F[0] if F else None)
        rows.append(dict(snapshot=r.insts[0], snapshot_day=r.snap_day, compared_on=day_of(when), sequence=seq,
                         net=r.net, sta=r.sta, cha=r.cha,
                         data_that_day=has_data, resp_epoch_start=r.start_raw, resp_epoch_end=r.end_raw,
                         resp_analog_1hz=r.amp[0], resp_analog_5hz=r.amp[1], resp_phase_deg=round(float(np.degrees(r.phase)), 1),
                         resp_sensor_datalogger=r.desc,
                         primary_source=p.rec.src if p else "", primary_epoch_start=p.rec.start_raw if p else "",
                         primary_confirmed_that_day=bool(p and when < p.rec.conf_until),
                         primary_analog_1hz=p.rec.amp[0] if p else np.nan, primary_analog_5hz=p.rec.amp[1] if p else np.nan,
                         primary_phase_deg=round(float(np.degrees(p.rec.phase)), 1) if p else np.nan,
                         max_diff_pct=round(100 * maxdiff(r.amp, p.rec.amp), 3) if p else np.nan,
                         start_diff_days=round((r.t0 - p.rec.t0) / 86400.0, 2) if p else np.nan,
                         verdict=verdict, merged_source=f.rec.src if f else "", merged_rule=f.role if f else "",
                         merged_epoch_start=iso(f.t0) if f else "", file=r.path))
    I = pd.DataFrame(rows)
    I.to_csv(HERE / "issue20_necis_resp.csv", index=False)

    # ---- summary
    print(f"\nchosen streams {len(C)}: {C.status.value_counts().to_dict()}; "
          f"with USE_SNAPSHOTS={not USE_SNAPSHOTS}: fully covered {int(((C.data_days > 0) & (C.days_without_response_alt == 0) & (C.days_excluded == 0)).sum())}")
    print(f"channel-days {len(D)}: covered {int((D.status == 'covered').sum())}, without response "
          f"{int((~D.status.isin(['covered', 'excluded'])).sum())} (boundary holes {int((D.status == 'boundary hole').sum())}, "
          f"excluded (X) {int((D.status == 'excluded').sum())}, partial "
          f"{int((D.status == 'partial').sum())}); with USE_SNAPSHOTS={not USE_SNAPSHOTS}: without response {int((~D.status_alt.isin(['covered', 'excluded'])).sum())}; "
          f"on snapshot epochs {int(((D.source != '') & ~D.source.isin(PRIMARY)).sum())}; unconfirmed "
          f"{int(((D.status == 'covered') & (D.confirmed_by == '')).sum())}; conflict {int((D.conflict != '').sum())}; "
          f"sign conflict {int((D.sign_conflict != '').sum())}; suspect {int((D.suspect != '').sum())}; "
          f"NECIS statements revised by NECIS later {int((D.revised != '').sum())}")
    print("covered channel-days by source:", D[D.status == "covered"].source.value_counts().to_dict())
    for k in SEQ:
        q = C[C[f"seq{k}_data_days"] > 0]
        print(f"sequence {k} {SEQ[k]}: {len(q)} chosen streams with data; " + "; ".join(
            f"{n} {int((q[f'seq{k}_{n}'] > 0).sum())} streams ({int(q[f'seq{k}_{n}'].sum())} stream-days)"
            for n in ("without_response", "excluded", "unconfirmed", "conflict", "sign_conflict")))
    print("merged epochs by rule:", pd.Series([p.role for ps in merged.values() for p in ps]).value_counts().to_dict())
    if len(I):
        print("issue20 verdicts:", I.verdict.str.replace(r"\d+(\.\d+)? ?(d|%)", "N\\2", regex=True).value_counts().to_dict())
    print("\nremove_response with the merged XML (2 h of one day; ratio = RMS 3-8 Hz against dividing by the full |H(5 Hz)|):")
    print(validate_remove_response(xml, build_xml(streams, merged_p)).to_string(index=False))


if __name__ == "__main__":
    buf = io.StringIO()

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)

        def flush(self):
            sys.__stdout__.flush()

    with redirect_stdout(Tee()):
        main()
    (HERE / "build_log.txt").write_text(buf.getvalue())
