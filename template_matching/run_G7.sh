#!/bin/bash
# GWANGYANG G7: template matching over 2012-12-20 .. 2026-10-06 (port of run_R5.sh, the Haenam renewal R5; nothing runs
# until it is called).
#   ./run_G7.sh settings          # print the run's settings (nothing written; the first templates/check/batch freezes them
#                                 # in logs/G7_5_20/run_settings.json)
#   ./run_G7.sh templates         # needs catalogs/templates/Gwangyang_templates_ref25.csv (catalogs/templates/build_template_catalog.py)
#   ./run_G7.sh check YEAR DOY    # scan one day + self-detection report (measures the time per day)
#   ./run_G7.sh batch [YEARS]     # the scan, year by year, one GPU process
#   ./run_G7.sh progress | catalog
#
# Settings (docs/decisions.csv):
#   templates   the focused 25 m set of the G5 relocation, picks from that run's exports only (G-D15, G-D16)
#   stations    the station table's codes within 60 km, one chosen sensor each (stations_G7_60km.csv; G-D4)
#   S           on N/E at every station, P on Z; each template's N/E turned into the scanned day's orientation (G-D15)
#   threshold   10 x MAD per configuration segment, no CC floor (Imsil D23; G-D6). Every 12 x MAD + 0.20 detection is also a
#               10 x MAD detection, so the Haenam R5 threshold is applied afterwards to the same scan for the comparison
#   unchanged from R5: band 5-20 Hz, windows, >= 4 stations, STEP 1, SNR >= 2 per channel with no ungated fallback
# Never run while detection (G4) or the cross-correlation of G5/G8 uses the GPU.
cd "$(dirname "$0")"
export FMF_RUN=G7_5_20 FMF_BAND=5,20 FMF_STATION_LIST=stations_G7_60km.csv FMF_TEMPLATE_CSV=Gwangyang_templates_ref25.csv
export FMF_WF_EXPORTS=G5 FMF_EXCLUDE_EPOCHS=none FMF_TURN_TEMPLATES=1 FMF_SEQUENCES=pre2021,2021,interval,2026,post2026
export FMF_SNR_MIN=2 FMF_SNR_FALLBACK=0
export FMF_MAD_K=10 FMF_ABS_CC_FLOOR=0
export FMF_PERIOD=2012-12-20,2026-10-06
export FMF_REFERENCE_CSV=$(cd "$(dirname "$0")/.." && pwd)/catalogs/reference_25m/focused_reference.csv   # the focused 25 m set (adopt_gwangyang.py)
YEARS=${G7_YEARS:-"2021 2026 2013 2014 2015 2016 2017 2018 2019 2020 2022 2023 2024 2025 2012"}   # the two episodes first
SET=logs/$FMF_RUN/run_settings.json
settings() {
  if [ "$1" != write ]; then env | grep -E "^(FMF_|MAD_|G7_)" | sort; [ -f $SET ] && { echo "frozen:"; cat $SET; }; return; fi
  mkdir -p logs/$FMF_RUN
  if [ ! -f $SET ]; then
    /home/msseo/miniforge3/bin/python - <<EOF
import json, os
keys = ["FMF_BAND", "FMF_STATION_LIST", "FMF_TEMPLATE_CSV", "FMF_WF_EXPORTS", "FMF_EXCLUDE_EPOCHS", "FMF_TURN_TEMPLATES",
        "FMF_SEQUENCES", "FMF_SNR_MIN", "FMF_SNR_FALLBACK", "FMF_MAD_K", "FMF_ABS_CC_FLOOR", "FMF_PERIOD", "FMF_REFERENCE_CSV", "MAD_SEGMENTED",
        "FMF_MAD_SUBSAMPLE", "FMF_STEP"]
env = dict(os.environ, MAD_SEGMENTED="1", FMF_MAD_SUBSAMPLE=os.environ.get("FMF_MAD_SUBSAMPLE", "8"), FMF_STEP=os.environ.get("FMF_STEP", "1"))
json.dump(dict(run="$FMF_RUN", settings={k: env[k] for k in keys},
               note="Gwangyang G7 (2026-10-15). Result-affecting settings; config.py loads them for any tool run with FMF_RUN set to this run and refuses conflicting values."),
          open("$SET", "w"), indent=2)
print("wrote $SET")
EOF
  fi
  cat $SET
}
case "$1" in
  settings) settings ;;
  batch) settings write >/dev/null; ./run_full_period.sh batch "${2:-$YEARS}" ;;
  templates) [ -f ../catalogs/templates/Gwangyang_templates_ref25.csv ] || { echo "no Gwangyang_templates_ref25.csv yet (G5 adoption first)"; exit 1; }; settings write >/dev/null; ./run_full_period.sh templates ;;
  check) settings write >/dev/null; ./run_full_period.sh "$@" ;;
  *) ./run_full_period.sh "$@" ;;
esac
