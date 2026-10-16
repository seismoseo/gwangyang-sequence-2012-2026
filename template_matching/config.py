"""Configuration for FMF GPU template matching on the Haenam sequences (2020 / 2026).

Ported from /home/msseo/works/FMF/buan_fmf (the Buan prototype). Same design, Haenam values:
  * templates = the 195-event JNU-augmented Haenam_joint relocation (registry product tm_templates_195);
    PICKS (P = SAC `a`, S = SAC `t0`) come from that run's per-event SAC bundles, WAVEFORMS are cut
    from the SAME continuous archive the scan reads (identical processing) -> self-detection ~1.0;
  * scan = the continuous symlink tree (registry product dl_continuous; KS 100 Hz + JU 200 Hz -> 100 Hz);
  * multi-year: the scan takes an explicit YEAR (the Haenam story spans 2020 -> 2026).
Every path outside this folder comes from the registry (registry/layout.csv). Nothing outside this folder is written.
"""
import os
import sys

import pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "registry" / "haenam.py").is_file()) / "registry")); import haenam   # registry paths

# ---- paths ----
FMF_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(FMF_ROOT)                                 # the folder above this one

# ---- one source of truth for a run's result-affecting settings (2026-10-01) -----------------------------------------------
# logs/<FMF_RUN>/run_settings.json holds the settings the run was scanned with. Any tool started with FMF_RUN set to that run
# inherits them (so figures and checks use exactly the scan's settings), and a conflicting value set in the environment is an
# error, not a silent override. (Figures for nb45/nb46 had been drawn with the default SNR gate 2.0 instead of the run's 1.2.)
def _load_run_settings():
    import json as _json
    run = os.environ.get("FMF_RUN", "")
    if not run:
        return
    p = os.path.join(FMF_ROOT, "logs", run, "run_settings.json")
    if not os.path.exists(p):
        return
    want = _json.load(open(p))["settings"]
    bad = {k: (os.environ[k], v) for k, v in want.items() if k in os.environ and os.environ[k] != v}
    if bad:
        raise SystemExit(f"[config] environment conflicts with {p}: " + ", ".join(f"{k}={a!r} (run: {b!r})" for k, (a, b) in bad.items()))
    for k, v in want.items():
        os.environ.setdefault(k, v)
_load_run_settings()
CONTINUOUS = haenam.path(None, "dl_continuous")                     # symlink tree (link_continuous.py)
STATIONS_CSV = haenam.path(None, "dl_station_table")
# Template catalog, named by its file name in FMF_TEMPLATE_CSV (a run's settings hold that name): haenam_templates.csv = the 195
# canonical events, haenam_templates_ref594.csv = the 594 events of the 25 m reference; the registry gives each file.
_TEMPLATE_PRODUCTS = {"haenam_templates.csv": "tm_templates_195", "haenam_templates_ref594.csv": "tm_templates_594",
                      "Haenam_templates_R4.csv": "tm_templates_R4",                  # renewal R5: the R4 focused 25 m set
                      "Gwangyang_templates_ref25.csv": "tm_templates_195"}           # GWANGYANG PORT: the focused 25 m set of the G5 relocation
TEMPLATE_CSV = haenam.path(None, _TEMPLATE_PRODUCTS[os.environ.get("FMF_TEMPLATE_CSV", "Gwangyang_templates_ref25.csv")])   # GWANGYANG PORT: default
# Per-event SAC bundles carrying the picks (a = P, t0 = S, reference time = origin, b = -30 s), searched in order: the joint
# 2020-2026 DL run (ref594 templates), the KCR joint relocation (KS/KG, 100 Hz) and its JNU augmentation (JU, 200 Hz), both
# in-project copies of the KCR run folders, and the 2026-only DL run (PyOcto picks).
# GWANGYANG PORT: the Haenam runs above do not exist here; the picks come only from the G5 relocation's own exports
# (registry wf_dl_2012_2026), the run whose 25 m set gives the templates
WF_EXPORTS = [haenam.path(None, "wf_dl_2012_2026")]
# renewal R5 (FMF_WF_EXPORTS=R4): the picks come only from the R4 relocation's own exports (the run whose 25 m set gives the
# templates), never from older runs
if os.environ.get("FMF_WF_EXPORTS", "") == "R4":
    WF_EXPORTS = [haenam.path(None, "wf_dl_2012_2026")]          # GWANGYANG PORT
DL_CATALOG = haenam.path(None, "dl_catalog")                         # analysis/haenam_catalog.py output
KMA_CSV = haenam.path(None, "kma_national_catalog")                  # KST times (KMA national catalogue of 2026-03-09)
KMA_2026_CSV = haenam.path(None, "kma_necis_domestic")              # GWANGYANG PORT: NECIS domestic catalog (UTC, origin_utc), to 2026-10-13

# ---- template subset ----
# TEMPLATE_SEQUENCES: which `sequence` values of haenam_templates.csv to use ("2020", "quiescence", "2026").
# A template is only buildable if its origin DAY exists in the continuous archive: today that is the 2026
# set (+ the 2022 quiescence event once 2022 data arrive; the 2020 set once 2020 data arrive). Events whose
# day files are missing are dropped with a message, never silently zero-filled.
TEMPLATE_SEQUENCES = tuple(os.environ.get("FMF_SEQUENCES", "pre2021,2021,interval,2026,post2026").split(","))   # GWANGYANG PORT: the eras of adopt_gwangyang.py
TEMPLATE_CONFIDENCE = "high_conf"

# ---- optional NEAR-SOURCE round (env-driven; default off) ----
# NEAR_SOURCE_KM=30 keeps only stations within that radius of the source (JNU + HAWA/MIAA/MOPB/HANB/JDO2/
# NPGA): a near-constant geometry where the simple 12xMAD threshold is clean (Buan lesson).
NEAR_SOURCE_RADIUS_KM = (lambda v: float(v) if v and float(v) > 0 else None)(os.environ.get("NEAR_SOURCE_KM", ""))
NEAR_SOURCE_CENTER = (35.0000, 127.7286)                     # GWANGYANG PORT: Gwangyang cluster centroid (lat, lon)
FMF_RUN = os.environ.get("FMF_RUN", "")                       # run name -> output suffix (e.g. full_period)
_SUF = (f"_near{int(NEAR_SOURCE_RADIUS_KM)}" if NEAR_SOURCE_RADIUS_KM else "") + (f"_{FMF_RUN}" if FMF_RUN else "")

# ---- full-period station policy (2026-09-22) ----
# STATION_LIST: explicit station set (network,station,horizontals). horizontals=1 -> Shelly S pass on N/E (JU:
# single verified orientation period); 0 -> S pass on Z (KS/KG borehole horizontals were rotated/reversed between
# epochs, verticals are all normal). Empty/missing file -> the original occurrence rule (MIN_TEMPLATE_OCCUR).
STATION_LIST = os.path.join(FMF_ROOT, os.environ.get("FMF_STATION_LIST", "stations_full_period.csv"))   # relative to this folder
SNR_MIN = float(os.environ.get("FMF_SNR_MIN", "2.0"))          # per-channel template gate: window RMS / pre-P noise RMS; 0 = off
NOISE_WIN = (6.0, 1.0)                                          # noise = [P-6 s, P-1 s] on the same channel (S-3 s if no P)
MIN_LIVE_P = int(os.environ.get("FMF_MIN_LIVE_P", "0"))       # at scan time (after the SNR mask): a template needs >= this many P (Z) channels (0: S-only templates allowed) ...
MIN_LIVE_CH = int(os.environ.get("FMF_MIN_LIVE_CH", "4"))      # ... and >= this many channels in total to be scanned that day
# SNR gate fallback: True (old runs) = a template with fewer than MIN_LIVE_CH channels over the gate runs ungated ("where
# affordable"); False (renewal R5, ISSUES #28, user 2026-10-14) = the gate always applies, and such a template is not scanned
SNR_FALLBACK = os.environ.get("FMF_SNR_FALLBACK", "1") not in ("", "0", "false", "False")
# Template S turned into the scanned day's orientation (renewal R5, ISSUES #27; tm_orientation.py). On in the renewal tree:
# P on Z and S on N/E at every station; the continuous data are never rotated. FMF_TURN_TEMPLATES=0 turns it off.
TURN_TEMPLATES = os.environ.get("FMF_TURN_TEMPLATES", "1") not in ("", "0", "false", "False")
# Sensor epochs that must never be mixed: templates inside the epoch drop the station; on scan days inside the epoch,
# templates from outside it get zero weight there (and vice versa).
EXCLUDE_EPOCHS = {("KS", "HANB"): [("2020-05-02", "2022-12-13")]}   # surface stand-in sensor (StationXML depth 0 m)
# renewal R5 (FMF_EXCLUDE_EPOCHS=none): HANB is read on its accelerometer (HG, one instrument over the whole window), so the
# broadband stand-in epoch above does not apply (renewal plan: "decision 29 is no longer needed")
if os.environ.get("FMF_EXCLUDE_EPOCHS", "") == "none":
    EXCLUDE_EPOCHS = {}


# Temporary station codes (station inventory decision A): registry/streams.csv rows whose code differs from the site,
# e.g. KS.JAEBT = JAEB and KS.JEDAT = JEDA for 2023-01-12..2023-07-03. The scan reads them as their site (scan._chan_files):
# the temporary code's day file on days without the site code's file, both files merged on the hand-over day.
def _temp_codes():
    import csv
    out = {}
    with open(haenam.ROOT / "registry" / "streams.csv") as fh:
        for r in csv.DictReader(fh):
            if r["sta"] != r["site"]:
                out.setdefault((r["net"], r["site"]), []).append(r["sta"])
    return out


TEMP_CODES = _temp_codes()

# Scan period (renewal R5: FMF_PERIOD=2018-01-31,2026-09-30, the user's choice of 2026-10-14): shelly.main skips days outside
# it; unset = every day with data
SCAN_PERIOD = tuple(os.environ["FMF_PERIOD"].split(",")) if os.environ.get("FMF_PERIOD") else None

TEMPLATE_NPZ = f"{FMF_ROOT}/templates_cache/haenam_templates{_SUF}.npz"
SHELLY_NPZ = f"{FMF_ROOT}/templates_cache/haenam_shelly{_SUF}.npz"

# ---- station / channel set ----
NETWORKS = ["KS", "KG", "JU"]          # KG has no continuous data here -> dropped by _cont_prefix
NET_REMAP = {}                         # SAC-pick/station-table net -> continuous net (none needed)
COMPONENTS = ["N", "E", "Z"]           # Z LAST (FMF idiom: vertical=last)
FS = 100.0                             # common sampling rate (Hz); JU 200 Hz is resampled
# Detection band (FMF prescribes none). 2 Hz kills microseisms; 15 Hz keeps the small-event energy well
# inside the 50 Hz Nyquist. Haenam's 20-km-deep sources at 4-100 km are in the same regime as Buan.
BANDPASS = tuple(float(x) for x in os.environ.get("FMF_BAND", "2,15").split(","))   # Hz; env FMF_BAND="5,20" for the 5-20 Hz run (the dt.cc band; band_test.py 2026-09-27)

# ---- template windows (Beauce single-length pass; picks are reused by shelly.py) ----
# Haenam S-P: ~2.5 s at JN09 (4 km epicentral, 20 km deep) to ~9 s at 100 km, so a 4 s window
# (1 s pre-pick) keeps the S window free of P on essentially every channel (Buan rationale).
WIN_LEN = 4.0
WIN_SAMPLES = int(round(WIN_LEN * FS))                      # 400 samples
P_PRE = 1.0                                                  # s before P (vertical window)
S_PRE = 1.0                                                  # s before S (horizontal windows)
PHASE_FOR_COMP = {"N": "S", "E": "S", "Z": "P"}
PRE_FOR_COMP = {"N": S_PRE, "E": S_PRE, "Z": P_PRE}
MIN_TEMPLATE_OCCUR = 8        # keep a station if it has picks in >= this many templates

# ---- scan ----
STEP = int(os.environ.get("FMF_STEP", "1"))                  # CC evaluated every STEP samples (1 = every sample, Buan policy)
SAMPLES_PER_DAY = int(round(86400 * FS))                    # 8_640_000

# ---- parallel data loading (the per-day bottleneck) ----
N_WORKERS = int(os.environ.get("FMF_WORKERS", "24"))        # shared 64-core box: stay polite (per process)

# ---- memory safety (never OOM) ----
# Device usage ~= DEVICE_OVERHEAD*(data + batch*n_corr*4) + templates (calibrated on Buan: 45.8 GB
# measured on 96 sta x 116 tpl). RTX PRO 6000 (98 GB) -> 45 GB ceiling leaves room for other users.
GPU_BUDGET_BYTES = int(float(os.environ.get("FMF_GPU_BUDGET", "45e9")))   # per process; lower when running years in parallel
DEVICE_OVERHEAD = 3.3

# ---- detection ----
MAD_K = float(os.environ.get("FMF_MAD_K", "12.0"))          # threshold = MAD_K * MAD(CC); GWANGYANG PORT: env-overridable (decision G-D6)
MAD_SUBSAMPLE = int(os.environ.get("FMF_MAD_SUBSAMPLE", "8"))   # median/MAD estimated on every k-th CC sample (8.6M -> 1.1M): same threshold, 8x faster
# Absolute network-CC floor + station gate (applied in build_catalog.py, exactly as the Buan notebooks
# did): network-CC MAD ~ 1/sqrt(n_stations), so on a large network 12xMAD collapses to noise level;
# the floor makes the threshold density-robust. Catalog = 12*MAD AND cc>=ABS_CC_FLOOR AND
# n_present>=MIN_PRESENT_STATIONS, declustered.
ABS_CC_FLOOR = float(os.environ.get("FMF_ABS_CC_FLOOR", "0.20"))   # GWANGYANG PORT: env-overridable (decision G-D6)
MIN_PRESENT_STATIONS = 4
# Config-segmented MAD (env MAD_SEGMENTED=1): threshold PER station-configuration segment (segmad.py);
# removes the station-onset/offset transition artifact. Boundaries from miniSEED HEADER coverage.
MAD_SEGMENTED = os.environ.get("MAD_SEGMENTED", "") not in ("", "0", "false", "False")
MAD_SEG_MERGE_S = 1800.0
MAD_SEG_MIN_S = 600.0
TEMPLATE_DURATION_S = WIN_LEN                               # declustering scales with this
DETECT_DIR = f"{FMF_ROOT}/catalogs/daily_shelly{_SUF}{'_seg' if MAD_SEGMENTED else ''}"
CATALOG_CSV = f"{FMF_ROOT}/catalogs/fmf_haenam_catalog{_SUF}{'_seg' if MAD_SEGMENTED else ''}.csv"

os.makedirs(DETECT_DIR, exist_ok=True)


def day_start(year, doy):
    from obspy import UTCDateTime
    return UTCDateTime(f"{year}-01-01") + (doy - 1) * 86400
