#!/bin/bash
# Consistent-component full-period FMF run at 5-20 Hz (2026-09-27 design, user rule: one S policy over all stations and the whole
# period; horizontals of temporally re-oriented stations are not used): P on Z everywhere; S on N/E at the 14 KS/KG stations with a
# stable orientation over 2020-2026 and at JN01/JN08/JN09; S on Z at MOPB, HANB, BGDB, JODB, IJDB (re-oriented horizontals). Band 5-20 Hz (dt.cc
# band); SNR gate 1.2, floor 0.20, >=4 present stations, segmented 12xMAD — as the earlier runs. Station list stations_full_period_H.csv.
# Outputs carry the suffix _full_period_5_20H (templates_cache, catalogs/daily_shelly_full_period_5_20H_seg, logs/full_period_5_20H).
cd "$(dirname "$0")"
export FMF_RUN=full_period_5_20H FMF_BAND=5,20 FMF_STATION_LIST=stations_full_period_H.csv
YEARS=${FMF_YEARS:-"2023 2020 2026 2021 2022 2024 2025"}
case "$1" in
  batch) ./run_full_period.sh batch "$YEARS" ;;
  *) ./run_full_period.sh "$@" ;;
esac
