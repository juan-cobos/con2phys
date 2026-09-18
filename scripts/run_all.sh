#!/usr/bin/env sh
# Run every answer script under src/Q*/ in question order, from the repo root.
#
#   sh scripts/run_all.sh              # all questions
#   sh scripts/run_all.sh Q4 Q12       # only these questions
#
# Each script runs as a module (uv run python -m src.QN.answer) so that the
# `from src...` imports resolve. Output is echoed to the terminal and saved to
# logs/QN_<name>.log. A failing script does not stop the rest; a summary of
# failures is printed at the end and the exit code is non-zero if any failed.

set -u

cd "$(dirname "$0")/.." || exit 1

LOG_DIR=logs
mkdir -p "$LOG_DIR"

# Force a non-interactive matplotlib backend so no figure window can block.
export MPLBACKEND=Agg

if [ "$#" -gt 0 ]; then
    QUESTIONS="$*"
else
    QUESTIONS=$(ls -d src/Q*/ | sed 's#src/##; s#/##' | sort -t Q -k 2 -n)
fi

failed=""
for q in $QUESTIONS; do
    for f in src/"$q"/answer*.py; do
        [ -e "$f" ] || { echo "!! no answer script in src/$q"; continue; }
        name=$(basename "$f" .py)
        module="src.$q.$name"
        log="$LOG_DIR/${q}_${name}.log"
        echo "================ $module ================"
        # Capture python's exit status despite the tee pipe (POSIX sh has no PIPESTATUS).
        { uv run python -m "$module" 2>&1; echo "$?" > "$LOG_DIR/.status"; } | tee "$log"
        status=$(cat "$LOG_DIR/.status")
        if [ "$status" -ne 0 ]; then
            echo "!! $module FAILED (see $log)"
            failed="$failed $module"
        fi
    done
done

echo
if [ -n "$failed" ]; then
    echo "Failed:$failed"
    exit 1
fi
echo "All answers ran successfully."
