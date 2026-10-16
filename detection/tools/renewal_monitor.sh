#!/bin/bash
# Continuous monitor of the R3 run: every 10 minutes one status line into logs/renewal_monitor.log; an ALERT line when the
# driver is gone before every year is done, an error appears in a log, or no new day was picked for 30 minutes.
# Restarts the read-ahead (prefetch_days.py) if it stopped while detection runs. Ends when the driver has ended.
cd "$(dirname "$0")/.."
PY=/home/msseo/miniforge3/bin/python
LOG=logs/renewal_monitor.log
prev=-1; last=$(date +%s); nerr_prev=0
while true; do
  line=$($PY -B tools/renewal_progress.py 2>&1)
  now=$(date +%s)
  done=$(sed -n 's/.*R3: \([0-9]*\)\/.*/\1/p' <<<"$line")
  [[ "$done" != "$prev" ]] && { prev=$done; last=$now; }
  echo "$line" >> $LOG
  nerr=$(grep -o "ERRORS [0-9]*" <<<"$line" | awk '{print $2}'); nerr=${nerr:-0}
  (( nerr > nerr_prev )) && { echo "ALERT new error lines: $line" >> $LOG; nerr_prev=$nerr; }
  locating=$(grep -c "locating \[" <<<"$line")
  (( now - last > 1800 && locating == 0 )) && echo "ALERT no new day picked for $(( (now - last) / 60 )) min: $line" >> $LOG
  if ! pgrep -u "$USER" -f "run_renewal_detection.sh years" >/dev/null; then
    echo "END driver finished: $line" >> $LOG; exit 0
  fi
  if pgrep -u "$USER" -f "pipeline/detection.py" >/dev/null && ! pgrep -u "$USER" -f "prefetch_days.py" >/dev/null; then
    nohup $PY -B tools/prefetch_days.py --ahead 3 >> logs/renewal_prefetch.log 2>&1 &
    echo "NOTE prefetch restarted $(date '+%m-%d %H:%M')" >> $LOG
  fi
  sleep 600
done
