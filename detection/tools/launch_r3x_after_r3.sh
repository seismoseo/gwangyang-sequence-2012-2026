#!/bin/bash
# Start R3x as soon as R3 has finished (the R3 driver has ended and logged "all years done").
cd "$(dirname "$0")/.."
while pgrep -u "$USER" -f "run_renewal_detection.sh years" >/dev/null; do sleep 60; done
grep -q "all years done" logs/renewal_status.txt || { echo "$(date '+%F %T') R3 driver ended without 'all years done': R3x NOT started" ; exit 1; }
echo "$(date '+%F %T') R3 finished; R3x prepare"
./run_r3x_detection.sh prepare && ./run_r3x_detection.sh run
echo "$(date '+%F %T') R3x launcher finished"
