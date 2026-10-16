"""The relocation set: which DL events go into HypoDD (renewal R4). One definition, read by
relocation/code/run_dl_hypodd.py (through event_sac_export.catalog_from_sum), detection/haenam_catalog.py (catalog column
relocation_set) and the R3 stage document.

  quality gate   detection/pipeline/config.py HAENAM_QC (sumio.apply_qc semantics): gap < 180 deg, RMS < 1.0 s,
                 ERH and ERZ < 5 km, at least 8 phases (user, 2026-10-13: the study's one quality rule)
  region         epicentral distance from the cluster center CENTER below RADIUS_KM (user, 2026-10-13: a 5 km radius
                 instead of the Haenam box, so the input is the source region by definition; the box, about +/-12 km,
                 let in deep seismicity elsewhere that the 25 m rule later removed, and one event excluded by name)
  same second    of two events in the same second, the first is kept (the relocation framework keys events by second)
  depth          at least MIN_DEPTH_KM (keeps the shallow quarry blasts out)

Order as in catalog_from_sum: gate -> region -> same-second rule -> depth.
"""
import numpy as np

CENTER = (35.0000, 127.7286)        # GWANGYANG PORT: = detection/pipeline/config.py HAENAM_EPICENTER (the Gwangyang centroid)
RADIUS_KM = 5.0
MIN_DEPTH_KM = 10.0


def epi_km(lat, lon, center=CENTER):
    """epicentral distance (km) from the cluster center, flat-earth (as config.HAENAM_EPICENTER distances elsewhere)"""
    return 111.2 * np.hypot(np.asarray(lat, float) - center[0], (np.asarray(lon, float) - center[1]) * np.cos(np.radians(center[0])))


def in_region(lat, lon):
    return epi_km(lat, lon) < RADIUS_KM
