#!/bin/bash
# After the R4 run ends ("[final]" in its log): the adoption (25 m set, focused set, plane), the R4 document, and the R5
# template catalog. No scan, no GPU. Each step's output goes to logs/after_r4.log; a failure stops the chain.
#   nohup detection/tools/after_r4.sh &      (waits for R4)
set -o pipefail
R=/home/msseo/works/27.Haenam_sequence_renewal
LOG=$R/detection/logs/after_r4.log
PY=/home/msseo/miniforge3/bin/python
until grep -q "\[final\]" $R/detection/logs/r4_reloc_fast2.log; do
  if grep -qE "Traceback|SystemExit" $R/detection/logs/r4_reloc_fast2.log; then echo "$(date '+%F %T') R4 FAILED: no after-R4 steps" >> $LOG; exit 1; fi
  sleep 60
done
sleep 60                                                  # let the driver finish writing haenam_dl_relocated.csv
{
  echo "=== $(date '+%F %T') R4 finished: adoption"
  cd $R && $PY -B relocation/code/adopt_r4.py --run-root relocation/hypodd_dl_renewal_R4 --apply || exit 1
  echo "=== $(date '+%F %T') R4 document"
  cd $R/docs/stages/R4_relocation && ./build.sh || exit 1
  echo "=== $(date '+%F %T') R5 template catalog"
  cd $R && $PY -B catalogs/templates/build_template_catalog_R4.py --apply || exit 1
  echo "=== $(date '+%F %T') after-R4 done"
} >> $LOG 2>&1
echo "$(date '+%F %T') AFTER-R4 DONE" >> $R/detection/logs/renewal_status.txt
