#!/bin/bash
# Renewal step R3: PhaseNet+ detection, association and HYPOINVERSE location over 2020-01-01 .. 2026-09-30, on the renewed
# station and stream tables (R2), with the pipeline settings unchanged (the frozen full-period settings).
#
#   ./run_renewal_detection.sh freeze        # record the parameter hash of this run (once, before the first year)
#   ./run_renewal_detection.sh test          # two test days: 2020-05-08 (DOY 129) and 2026-08-22 (DOY 234)
#   ./run_renewal_detection.sh years         # parallel chains (env CHAINS; default "2020,2026 2021,2025 2022,2023 2024"): detect, then locate
#   ./run_renewal_detection.sh status        # per-year counts (tools/run_status.py)
# Each year is resumable: detection skips days whose picks file exists. Logs: logs/renewal_<year>_{detect,locate}.log.
cd "$(dirname "$0")"
PY=/home/msseo/miniforge3/bin/python
export WORKERS=${WORKERS:-4} NICE=${NICE:-10}
LAST_DOY_2026=279                                              # GWANGYANG PORT: 2026-10-06, the end of the study window
PARAMS_JSON=../relocation/provenance/adoption/renewal_run_params.json
PARAM_FILES="pipeline/config.py pipeline/core.py pipeline/pick_augmentation.py station_table/stations.csv /home/msseo/works/14.EQNet/EQNet/docs/model_phasenet_plus/model_99.pth"
hash_now() { sha256sum $PARAM_FILES | sha256sum | cut -c1-16; }
check_hash() {
  [[ -f $PARAMS_JSON ]] || { echo "no $PARAMS_JSON: run '$0 freeze' first"; exit 2; }
  ref=$($PY -c "import json;print(json.load(open('$PARAMS_JSON'))['hash'])")
  [[ "$(hash_now)" == "$ref" ]] || { echo "ABORT: parameter hash $(hash_now) != frozen $ref"; exit 2; }
}
chain() {                                                      # chain CORES YEAR...: the chain runs on its own CPU cores
  cores=$1; shift                                              # (torch sizes its thread pool to the affinity: no oversubscription)
  for y in "$@"; do
    days=1-366; [[ $y == 2026 ]] && days=1-$LAST_DOY_2026
    taskset -c "$cores" env OMP_WAIT_POLICY=PASSIVE ./run_haenam.sh detect "$y" "$days" run > "logs/renewal_${y}_detect.log" 2>&1
    taskset -c "$cores" ./run_haenam.sh locate "$y" run > "logs/renewal_${y}_locate.log" 2>&1
    echo "$(date '+%F %T') year $y done (detect rc ok if picks exist; see logs)" >> logs/renewal_status.txt
  done
}
case "$1" in
  freeze)
    [[ -f $PARAMS_JSON ]] && { echo "already frozen: $PARAMS_JSON"; exit 0; }
    $PY - "$(hash_now)" "$PARAMS_JSON" <<'PYEOF'
import json, sys, hashlib, datetime
sys.path.insert(0, "pipeline"); import config
d = dict(run="Gwangyang G4 (GWANGYANG PORT of renewal R3)", hash=sys.argv[1], frozen=datetime.datetime.now().isoformat(timespec="seconds"),
         window="2012-12-20 .. 2026-10-06", station_table="station_table/stations.csv (build_stations_renewal.py)",
         PNPLUS_MIN_PROB=config.PNPLUS_MIN_PROB, HAENAM_QC=config.HAENAM_QC, HAENAM_QC_A=config.HAENAM_QC_A, HAENAM_QC_B=config.HAENAM_QC_B,
         HYP_DIS=config.HYP_DIS, HYP_ZTR=config.HYP_ZTR, CHANNEL_POLICY=config.CHANNEL_POLICY, SAMPLING_RATE=config.SAMPLING_RATE,
         ANTIALIAS=config.ANTIALIAS, weights_sha256=hashlib.sha256(open(config.EQNET_WEIGHTS, "rb").read()).hexdigest())
json.dump(d, open(sys.argv[2], "w"), indent=1, default=str); print("frozen ->", sys.argv[2], d["hash"])
PYEOF
    ;;
  test)
    check_hash
    ./run_haenam.sh detect 2021 239-240 run && ./run_haenam.sh detect 2026 1-1 run && ./run_haenam.sh detect 2013 200-200 run ;;   # GWANGYANG PORT: G3 smoke days
  years)
    check_hash
    CHAINS=${CHAINS:-"2021,2013,2014 2026,2012,2015,2016 2017,2018,2019,2020 2022,2023,2024,2025"}   # GWANGYANG PORT: balanced by station-days
    echo "$CHAINS" > logs/renewal_chains.txt
    echo "$(date '+%F %T') start: chains [$CHAINS] on cores [${CORESETS:-0-15 16-31 32-47 48-63}], WORKERS=$WORKERS, OMP_WAIT_POLICY=PASSIVE" >> logs/renewal_status.txt
    CORESETS=(${CORESETS:-0-15 16-31 32-47 48-63})              # one core set per chain
    i=0; for c in $CHAINS; do chain "${CORESETS[$i]}" ${c//,/ } & i=$((i+1)); done
    wait
    echo "$(date '+%F %T') all years done" >> logs/renewal_status.txt ;;
  status)
    $PY tools/run_status.py --years 2012-2026 ;;
  *) sed -n 2,10p "$0" ;;
esac
