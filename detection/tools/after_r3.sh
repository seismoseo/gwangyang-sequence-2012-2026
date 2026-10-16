#!/bin/bash
# After R3 (replaces tools/launch_r3x_after_r3.sh, 2026-10-13): as soon as the R3 driver has ended and logged "all years
# done", in this order:
#   1. the R3 catalog (catalogs/Haenam_DL_catalog_2020-2026.csv), while station_table/stations.csv is still R3's table;
#   2. R3's station table saved as stations_r3.csv, then R3x (prepare adds the predecessor codes to stations.csv; run is
#      the ~8 h detection of 2018-01-31 .. 2019-12-31) in the background, exactly as the earlier launcher did;
#   3. beside R3x, the light R3 checks: the association test for every year (8 threads each, one year at a time),
#      notebook 22 executed in place on the new catalog, and the R3 document built (it runs the old-new comparison).
# Every step logs to logs/after_r3_*.log and writes one line to logs/renewal_status.txt.
cd "$(dirname "$0")/.."
PY=/home/msseo/miniforge3/bin/python
say() { echo "$(date '+%F %T') $*" >> logs/renewal_status.txt; }
while pgrep -u "$USER" -f "run_renewal_detection.sh years" >/dev/null; do sleep 60; done
grep -q "all years done" logs/renewal_status.txt || { say "R3 driver ended without 'all years done': after_r3 stopped (no catalog, no R3x)"; exit 1; }

./run_haenam.sh catalog run > logs/after_r3_catalog.log 2>&1 && say "R3 catalog written (after_r3_catalog.log)" || say "R3 catalog FAILED (after_r3_catalog.log)"
[[ -f station_table/stations_r3.csv ]] || cp -p station_table/stations.csv station_table/stations_r3.csv

( say "R3x prepare"; ./run_r3x_detection.sh prepare && ./run_r3x_detection.sh run; say "R3x launcher finished" ) > logs/r3x_launch.log 2>&1 &

for y in 2020 2021 2022 2023 2024 2025 2026; do
  nice -n 10 $PY -B tools/ablation_association.py --year $y --threads 8 >> logs/after_r3_ablation.log 2>&1 || say "association test $y FAILED"
done
say "R3 association test done (after_r3_ablation.log)"
( cd ../analysis/seismicity && nice -n 10 /home/msseo/miniforge3/bin/jupyter nbconvert --to notebook --execute --inplace 22_full_catalog_overview.ipynb ) > logs/after_r3_nb22.log 2>&1 \
  && say "notebook 22 executed on the R3 catalog" || say "notebook 22 FAILED (after_r3_nb22.log)"
../docs/stages/R3_detection/build.sh > logs/after_r3_doc.log 2>&1 && say "R3 document built" || say "R3 document build FAILED (after_r3_doc.log)"
wait
