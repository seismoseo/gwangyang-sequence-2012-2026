#!/bin/bash
# Renewal R5: template matching on the renewed inputs (prepared 2026-10-14; nothing runs until it is called).
#   ./run_R5.sh settings          # print the run's settings (nothing written; the first templates/check/batch freezes them
#                                 # in logs/R5_5_20/run_settings.json)
#   ./run_R5.sh templates         # needs catalogs/templates/Haenam_templates_R4.csv (after R4's adoption)
#   ./run_R5.sh check YEAR DOY    # scan one day + self-detection report (measures the time per day)
#   ./run_R5.sh batch [YEARS]     # the scan, year by year, one GPU process (as the old runs)
#   ./run_R5.sh progress | catalog
#
# What changes from the old 5-20 Hz run (run_5_20H.sh), all decided:
#   templates   the R4 focused 25 m set (Haenam_templates_R4.csv), picks from the R4 exports only (FMF_WF_EXPORTS=R4)
#   stations    the renewal's stream table, one chosen sensor per station (stations_R5.csv; the user chooses the scan set)
#   S           on N/E at every station, P on Z (ISSUES #27); each template's N/E turned into the scanned day's orientation
#               (tm_orientation.py, FMF_TURN_TEMPLATES=1; tested on real data: logs/test_tm_orientation.csv)
#   HANB        read on its accelerometer, so the broadband stand-in epoch exclusion does not apply (FMF_EXCLUDE_EPOCHS=none)
#   grid        every channel on the common 10 ms sample grid (ISSUES #34, already in scan/templates/similarity)
# Unchanged: band 5-20 Hz, windows, segmented 12 x MAD with the 0.20 network-CC floor and >= 4 stations, STEP 1.
#
# The user's choices (2026-10-14): stations within 50 km (stations_R5_50km.csv, 32); SNR >= 2 per channel with no ungated
# fallback (ISSUES #28); 12 x MAD + 0.20 floor (unchanged); period 2018-01-31 .. 2026-09-30.
cd "$(dirname "$0")"
export FMF_RUN=R5_5_20 FMF_BAND=5,20 FMF_STATION_LIST=stations_R5_50km.csv FMF_TEMPLATE_CSV=Haenam_templates_R4.csv
export FMF_WF_EXPORTS=R4 FMF_EXCLUDE_EPOCHS=none FMF_TURN_TEMPLATES=1 FMF_SEQUENCES=2018-2019,2020,quiescence,2026
export FMF_SNR_MIN=2 FMF_SNR_FALLBACK=0              # ISSUES #28 (user, 2026-10-14)
export FMF_PERIOD=2018-01-31,2026-09-30              # the scan period (user, 2026-10-14); shelly.main skips other days
YEARS=${R5_YEARS:-"2026 2020 2018 2019 2021 2022 2023 2024 2025"}   # 2018-01-31 .. 2026-09-30, the sequences first
SET=logs/$FMF_RUN/run_settings.json
settings() {
  if [ "$1" != write ]; then env | grep -E "^(FMF_|MAD_|R5_)" | sort; [ -f $SET ] && { echo "frozen:"; cat $SET; }; return; fi
  mkdir -p logs/$FMF_RUN
  if [ ! -f $SET ]; then
    /home/msseo/miniforge3/bin/python - <<EOF
import json, os
keys = ["FMF_BAND", "FMF_STATION_LIST", "FMF_TEMPLATE_CSV", "FMF_WF_EXPORTS", "FMF_EXCLUDE_EPOCHS", "FMF_TURN_TEMPLATES",
        "FMF_SEQUENCES", "FMF_SNR_MIN", "FMF_SNR_FALLBACK", "FMF_PERIOD", "MAD_SEGMENTED", "FMF_MAD_SUBSAMPLE", "FMF_STEP"]
env = dict(os.environ, MAD_SEGMENTED="1", FMF_MAD_SUBSAMPLE=os.environ.get("FMF_MAD_SUBSAMPLE", "8"), FMF_STEP=os.environ.get("FMF_STEP", "1"))
json.dump(dict(run="$FMF_RUN", settings={k: env[k] for k in keys},
               note="Renewal R5 (2026-10-14). Result-affecting settings; config.py loads them for any tool run with FMF_RUN set to this run and refuses conflicting values."),
          open("$SET", "w"), indent=2)
print("wrote $SET")
EOF
  fi
  cat $SET
}
case "$1" in
  settings) settings ;;
  batch) settings write >/dev/null; ./run_full_period.sh batch "${2:-$YEARS}" ;;
  templates) [ -f ../catalogs/templates/Haenam_templates_R4.csv ] || { echo "no Haenam_templates_R4.csv yet (R4 adoption first)"; exit 1; }; settings write >/dev/null; ./run_full_period.sh templates ;;
  check) settings write >/dev/null; ./run_full_period.sh "$@" ;;
  *) ./run_full_period.sh "$@" ;;
esac
