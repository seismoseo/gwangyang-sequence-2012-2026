#!/bin/bash
# Renewal R4 relocation (user, 2026-10-14: proceed overnight; relocate 2018-2026 with the same selection as 2020-2026).
#   input   relocation/code/relocation_selection.py: quality gate, within 5 km of the cluster center, depth >= 10 km,
#           first of a same-second pair (run_dl_hypodd.py --radius-region --min-depth 10)
#   method  as the old full-period run (EXT0922: open WDCC, 1000 bootstrap replicas, blocked pair order, 4000-pair chunks)
#           except S differential times on T (--s-on-t; registry/orientation.py)
# Steps: wait for R3x ("R3x done"), a 20-event test run (own run root, no bootstrap) with checks, then the full run.
cd "$(dirname "$0")/.."
say() { echo "$(date '+%F %T') $*" >> logs/renewal_status.txt; }
until grep -q "R3x done" logs/renewal_status.txt; do sleep 60; done
export PYTHONUNBUFFERED=1 XCORR_TRACE_CACHE=30000
ARGS="--years 2018-2026 --radius-region --min-depth 10 --open-wdcc --s-on-t --cores 24 --blocked-pairs 45 --xcorr-chunk 4000"

say "R4 test run (20 events, no bootstrap) -> relocation/hypodd_r4_test"
nice -n 5 ./run_haenam.sh dl-reloc $ARGS --max-events 20 --n-boot 0 --run-root ../relocation/hypodd_r4_test run > logs/r4_test.log 2>&1
T=../relocation/hypodd_r4_test
nT=$(find $T -name "*T.sac" 2>/dev/null | wc -l); nZ=$(find $T -name "*Z.sac" 2>/dev/null | wc -l)
nS=$(find $T -path "*dt.cc_S*" -type f -size +0 2>/dev/null | wc -l)
loc=$(ls $T/haenam_dl_relocated.csv 2>/dev/null | wc -l)
if [[ $nT -gt 0 && $nS -gt 0 && $loc -eq 1 ]] && ! grep -q "Traceback" logs/r4_test.log; then
  say "R4 test passed: $nZ Z and $nT T traces, $nS non-empty S dt.cc files, relocated table written"
else
  say "R4 test FAILED (Z $nZ, T $nT, S dt.cc $nS, table $loc; logs/r4_test.log): full run NOT started"; exit 1
fi

say "R4 full run start -> relocation/hypodd_dl_renewal_R4"
nice -n 5 ./run_haenam.sh dl-reloc $ARGS --n-boot 1000 --run-root ../relocation/hypodd_dl_renewal_R4 run > logs/r4_reloc.log 2>&1 \
  && say "R4 full run done (logs/r4_reloc.log)" || say "R4 full run FAILED (logs/r4_reloc.log)"
