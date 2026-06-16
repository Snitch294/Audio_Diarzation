#!/usr/bin/env bash
# SPOVNOB one-command runner.
#
#   ./run.sh <videos_dir> [batch_name]
#
# Drop your interview videos into <videos_dir> and run this. It does, in order:
#   1. activate the pinned .venv + CUDA-13 LD_LIBRARY_PATH (env.sh)
#   2. collect the videos in sorted (canonical) order
#   3. clicking: reuse session/<batch>/clicks.json if present, else open the
#      clicking UI on the FIRST video so you mark the target speaker, then
#      continue automatically once you Export
#   4. full pipeline: environment gate -> Layer 0 -> 1 -> 2 -> 3
#   5. audit dashboard (single self-contained HTML)
#
# The first video (alphabetically) is file_index=0 — the one you click on, and
# the one that must contain the target (and the interviewer, for an anti-click).
set -euo pipefail

# Resolve the videos dir against your CURRENT directory before we cd into the
# project, so a relative path like ./my_videos works from anywhere.
RAW_VIDEO_DIR="${1:?usage: ./run.sh <videos_dir> [batch_name]}"
VIDEO_DIR="$(cd "$RAW_VIDEO_DIR" 2>/dev/null && pwd)" \
  || { echo "ERROR: videos dir not found: $RAW_VIDEO_DIR"; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
# shellcheck disable=SC1091
source env.sh

STORE="${SPOVNOB_MODEL_STORE:-/home/user1/model_store}"
BATCH="${2:-batch_$(date +%Y%m%d_%H%M%S)}"
WORK="session/${BATCH}"
MANIFEST="session/${BATCH}.manifest.jsonl"
CLICKS="${WORK}/clicks.json"
DASH="session/${BATCH}_audit.html"

mkdir -p "$WORK"

# 1. Collect videos in sorted (canonical) order.
mapfile -t VIDEOS < <(find -L "$VIDEO_DIR" -maxdepth 1 -type f \
  \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' \
     -o -iname '*.avi' -o -iname '*.webm' \) | sort)
[ "${#VIDEOS[@]}" -gt 0 ] || { echo "ERROR: no videos found in $VIDEO_DIR"; exit 1; }
echo "Batch:  $BATCH"
echo "Videos (canonical order; [0] is the clicking target):"
i=0; for v in "${VIDEOS[@]}"; do echo "  [$i] $v"; i=$((i+1)); done

# 2. Clicks: reuse if present, else launch the UI for the operator.
if [ -f "$CLICKS" ]; then
  echo "Using existing clicks: $CLICKS"
else
  echo
  echo ">> No clicks yet — opening the clicking UI on [0]."
  echo ">> In the browser: click the TARGET speaker's face (and optionally the"
  echo ">> interviewer for the anti-click), then press 'Export clicks.json'."
  python3 click_ui.py "${VIDEOS[0]}" --model-store "$STORE" \
    --work-dir "$WORK" --port 5050 > "$WORK/click_ui.log" 2>&1 &
  UI_PID=$!
  trap 'kill "$UI_PID" 2>/dev/null || true' EXIT
  echo -n ">> Waiting for clicks.json to be exported"
  while [ ! -f "$CLICKS" ]; do
    if ! kill -0 "$UI_PID" 2>/dev/null; then
      echo; echo "ERROR: click UI exited before export. See $WORK/click_ui.log"; exit 1
    fi
    sleep 2; echo -n "."
  done
  echo " captured."
  read -r -p ">> Press Enter to run the pipeline (Ctrl-C to keep editing clicks)... " _ || true
  kill "$UI_PID" 2>/dev/null || true
  trap - EXIT
fi

# 3. Full pipeline: gate -> Layers 0-3.
echo
echo ">> Running pipeline (gate -> Layers 0-3)..."
python3 pipeline_runner.py --run --videos "${VIDEOS[@]}" \
  --clicks "$CLICKS" --work-dir "$WORK" --model-store "$STORE" \
  --manifest "$MANIFEST" --operator "${USER:-operator}"

# 4. Audit dashboard.
echo
echo ">> Building audit dashboard..."
python3 audit_visualizer.py "$MANIFEST" --audio "$WORK" --out "$DASH"

echo
echo "DONE."
echo "  Clean audio:  $WORK/layer3/clean/"
echo "  Summary JSON: $WORK/pipeline_output.json"
echo "  Dashboard:    $DASH"
