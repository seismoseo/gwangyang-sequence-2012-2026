#!/bin/bash
# SUPERSEDED 2026-10-15 18:05 by necis_refetch_episodes.py (guards against the downloader selecting every station); jobs 1-3 ran here.
# GWANGYANG (decision G-D13): re-fetch from NECIS the KS components missing on MARA inside the 2021 episode
# (docs/station_inventory/tables/component_audit_2021_2026.csv and the whole-day gaps of the same audit, 2026-10-15).
# Only the missing components are requested: necis_refetch is the first source of link_continuous.py, so anything
# requested here replaces the MARA file of that channel and day.
# Not requested: 2021-08-27 BOSB HHZ and GMNA HGZ (asked 2026-10-15 09:47, NECIS has none); KG WDL 2021-07-17/18 (KIGAM, not
# NECIS); KS HWJA 2026-01-03..01-29 (a station outage: the NECIS archive has no HWJA day file from 2026-01-03 to 2026-05-09).
#   ./necis_refetch_episodes.sh            # list the jobs
#   ./necis_refetch_episodes.sh run        # run them one after another (one NECIS login per job)
cd "$(dirname "$0")/../.."
OUT=data/continuous/necis_refetch/continuous
REQ=data/continuous/necis_refetch/requests
LOG=logs/necis_refetch_episodes_20261015.log
JOBS="2021-07-02|KUJA|HGZ,HGN,HGE|2
2021-07-10|MALA|HGZ,HGN,HGE|2
2021-08-08|JINA|HGZ|5
2021-08-20|DUSB|HHZ|5
2021-08-20|GWYB|HHZ,HHN,HHE|2
2021-08-24|HAMB,DUSB|HHZ,HHN,HHE|2
2021-08-24|WICA,JNYA,GDDB|HGZ,HGN,HGE|2
2021-08-26|JAGA,NAJA|HGZ|5
2021-08-26|JEU2|ELZ|5
2021-08-27|BAU|ELZ,ELN,ELE|2
2021-08-28|GWYB|HHZ|5
2021-08-28|BAU|ELZ|5
2021-08-28|BLLA,GMNA,NAHA,SCHA,YESA|HGZ|5"
mkdir -p $REQ logs
n=0
while IFS="|" read -r day stas chans batch; do
  n=$((n+1)); tag=$(echo "$day" | tr -d -)_$(echo "$chans" | cut -c1-2)_$n
  csv=$REQ/stations_${tag}.csv
  /home/msseo/miniforge3/bin/python - "$stas" "$csv" <<'EOF'
import sys, pandas as pd
st = pd.read_csv("detection/station_table/stations.csv").set_index("Code")
rows = [dict(network="KS", station=s, startdate="1/1/12", latitude=st.Latitude[s], longitude=st.Longitude[s], datetime_start="2012-01-01")
        for s in sys.argv[1].split(",")]
pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
EOF
  echo "job $n: $day $stas $chans (batch $batch) -> $csv"
  if [ "$1" = run ]; then
    CONTINUOUS_DIR=$(pwd)/$OUT STATION_CSV=$(pwd)/$csv CHANNELS=$chans BATCH_SIZE=$batch \
      /home/msseo/works/Claude/archive_necis.sh "$day" "$day" >> $LOG 2>&1
    echo "job $n exit $? $(date '+%F %T')" >> $LOG
  fi
done <<< "$JOBS"
[ "$1" = run ] && echo "ALL_JOBS_DONE $(date '+%F %T')" >> $LOG
