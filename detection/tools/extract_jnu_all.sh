#!/bin/bash
# Extract ALL days of the three near-source JNU stations from their per-station tars
# (data/jnu/JN01.tar, JN08.tar, JN09.tar: 2025-12-04 -> 2026-08-31 04:09 UTC) into
# data/jnu/mseed/JU/<STA>/ — the same layout the Earthworm archive uses, so tools/link_continuous.py
# picks every day up. extract_jnu_days.sh only pulled DOY 226-243 (the 2026 sequence); the
# pre-sequence quiescence scan (2025-12 -> 2026-08) and the FMF year scan need the rest.
#
# Disk: ~62 GB (3 stations x ~270 days x 3 channels x ~23 MB). tar is read once per station,
# sequentially (the tars sit on the same disk). Existing files are skipped (tar -k).
#
#   ./tools/extract_jnu_all.sh            # DRY RUN: lists tar members per station, extracts nothing
#   ./tools/extract_jnu_all.sh run        # extract (nohup ... > logs/extract_jnu_all.log 2>&1 &)
#   then: python tools/link_continuous.py --apply
PROJECT_ROOT=$(python3 -c "import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path('${BASH_SOURCE[0]}').resolve().parents if (p / 'registry' / 'haenam.py').is_file()) / 'registry')); import haenam; print(haenam.ROOT)") || exit 2   # the project root: registry/haenam.py found upward from this script (decision 25)
set -u
JNU=${PROJECT_ROOT}/data/jnu
OUT=$JNU/mseed/JU
MODE=${1:-dry}
for STA in JN01 JN08 JN09; do
  n_have=$(ls "$OUT/$STA" 2>/dev/null | wc -l)
  if [[ "$MODE" == "run" ]]; then
    echo "[$(date '+%m-%d %H:%M')] $STA: extracting (tar read once; $n_have files already present)"
  else
    n_tar=$(tar -tf "$JNU/$STA.tar" | grep -c "$STA.JU.00.HH")     # dry run only: reads the whole tar
    echo "[$(date '+%m-%d %H:%M')] $STA: $n_tar channel-day files in tar, $n_have already extracted"
  fi
  if [[ "$MODE" == "run" ]]; then
    mkdir -p "$OUT"
    tar -xkf "$JNU/$STA.tar" -C "$OUT" --wildcards "$STA/$STA.JU.00.HH?.*" 2>&1 \
      | grep -v "Cannot open: File exists" | grep -v "Exiting with failure" | head -5
    echo "[$(date '+%m-%d %H:%M')] $STA done: $(ls "$OUT/$STA" | wc -l) channel-day files"
  fi
done
[[ "$MODE" == "run" ]] || echo "(dry run — pass 'run' to extract)"
