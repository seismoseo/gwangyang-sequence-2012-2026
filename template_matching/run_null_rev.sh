#!/bin/bash
# Null test (2026-09-30): the refined run's templates reversed in time (templates_cache/haenam_shelly_full_period_5_20H_rev.npz),
# same stations, band, moveouts, weights, SNR gate and detection rule; scanned on 10 busy days (2020 DOY 117-126, the 2020 episode)
# and 10 quiet days (2020 DOY 1-10). Detections by reversed templates estimate chance detections at the same rule.
cd "$(dirname "$0")"
export FMF_RUN=full_period_5_20H_rev FMF_BAND=5,20 FMF_STATION_LIST=stations_full_period_H.csv MAD_SEGMENTED=1 FMF_TEMPLATE_CSV=haenam_templates_ref594.csv FMF_SNR_MIN=1.2 FMF_MAD_SUBSAMPLE=8 FMF_STEP=1
export FMF_GPU_LOCK=$(pwd)/logs/gpu.lock FMF_WORKERS=8 FMF_GPU_BUDGET=12e9 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/home/msseo/miniforge3/envs/fmf/bin/python; mkdir -p logs/full_period_5_20H_rev
nice -n 15 $PY shelly.py 2020 117 126 >> logs/full_period_5_20H_rev/scan.log 2>&1
nice -n 15 $PY shelly.py 2020 1 10 >> logs/full_period_5_20H_rev/scan.log 2>&1
echo "null done $(date)" >> logs/full_period_5_20H_rev/scan.log
