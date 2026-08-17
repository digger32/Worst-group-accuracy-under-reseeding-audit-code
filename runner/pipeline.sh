#!/usr/bin/env bash
# A5 pipeline -- one command per stage, run from the PROJECT ROOT inside tmux:
#   tmux new -s a5   ->   bash runner/pipeline.sh <stage>   ->   Ctrl-b d
#
# Stages, in order:
#   wheels      build ~/wheelhouse once WITH network (only time the net is needed)
#   env         create .venv OFFLINE from the wheelhouse + PIP_CONSTRAINT guard
#   check       probe every dataset/checkpoint mirror -- downloads NOTHING
#   probe       stream a few CelebA rows to see WHICH split strategy is possible
#   data        download raw corpora + torchvision backbone weights
#   prep        decode EVERY image once under guard -> uint8 memmap + quarantine
#   selftest    gate unit-test on synthetic clean/dirty runs
#   smoke       1 unit, tiny subset, full chain run->aggregate->stats->gate
#   microfast   cheapest unit per method: measure wall_s, choose the path
#   microheavy  the HEAVIEST unit: measure peak RSS and host low-water RAM
#   tune        equal-trial HPO per (dataset,method) on seed 0 -> configs/tuned.yaml
#   pilot       2 seeds x 2 datasets x all methods
#   triage      rank repeated log lines from pilot -> candidate console filter
#   full        complete grid, resume ENABLED   (JOBS=N for within-host concurrency)
#   final       fresh outdir, --no-resume, then aggregate+stats+gate
#   stats|gate  re-run on an existing outdir (OUT=runs/... override)
#
# LOGS. Every stage is tee'd RAW and complete to logs/<stage>_<utc>.log; nothing is
# ever discarded. configs/log_filter.txt only hides lines from the CONSOLE, and it
# stays empty until `triage` produces evidence about what is actually noise.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python3}
JOBS=${JOBS:-1}   # concurrent units on THIS host; take it from microheavy, never guess
STAGE="${1:?usage: pipeline.sh <wheels|env|check|probe|data|prep|selftest|smoke|microfast|microheavy|tune|pilot|triage|full|final|stats|gate>}"
mkdir -p logs
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RAW="logs/${STAGE}_${STAMP}.log"
FILTER=configs/log_filter.txt
[ -f "$FILTER" ] || : > "$FILTER"
grep -v '^[[:space:]]*$' "$FILTER" | grep -v '^#' > /tmp/a5_filter.$$ || true

# run <cmd...> : full output to $RAW, filtered copy to the console, real exit code
run() {
  "$@" 2>&1 | tee -a "$RAW" | { grep -v -f /tmp/a5_filter.$$ || true; }
  return "${PIPESTATUS[0]}"
}
chain() {  # aggregate -> stats -> gate on $1
  run $PY scripts/aggregate.py "$1" && run $PY scripts/stats.py "$1" && run $PY scripts/review_gate.py "$1"
}
trap 'rm -f /tmp/a5_filter.$$' EXIT
echo "[pipeline] stage=$STAGE raw log -> $RAW" | tee -a "$RAW"

case "$STAGE" in
  wheels)
    run bash scripts/build_wheelhouse.sh ;;
  env)
    run bash scripts/make_venv.sh ;;
  check)
    run $PY scripts/download_data.py --check ;;
  probe)
    run $PY scripts/probe_celeba.py ;;
  data)
    run $PY scripts/download_data.py ;;
  prep)
    run $PY scripts/prepare_data.py ;;
  selftest)
    run $PY scripts/gate_selftest.py ;;

  smoke)
    run $PY scripts/gate_selftest.py || exit 1
    OUT=runs/smoke; rm -rf "$OUT"
    run $PY runner/bench_runner.py --outdir "$OUT" --datasets waterbirds \
        --methods erm --max-seeds 1 --timeout-s 900
    run $PY scripts/aggregate.py "$OUT"
    run $PY scripts/stats.py "$OUT" || true      # 1 unit: omnibus legitimately degenerate
    run $PY scripts/review_gate.py "$OUT" || true # smoke is resume-mode: A1 FAIL is EXPECTED
    echo "[smoke] done -- a gate FAIL on A1/B1 here is EXPECTED" | tee -a "$RAW" ;;

  # ---- two micro slices; both measure, neither guesses -------------------- #
  microfast)
    OUT=runs/micro_fast; rm -rf "$OUT"
    run $PY runner/bench_runner.py --outdir "$OUT" --datasets waterbirds \
        --max-seeds 1 --timeout-s 3600
    run $PY scripts/micro_report.py "$OUT" --mode fast ;;
  microheavy)
    OUT=runs/micro_heavy; rm -rf "$OUT"
    run $PY scripts/micro_report.py --emit-heavy-units > /tmp/a5_heavy.$$
    while read -r U; do
      [ -n "$U" ] && run $PY runner/bench_runner.py --outdir "$OUT" --only "$U" --timeout-s 10800
    done < /tmp/a5_heavy.$$
    rm -f /tmp/a5_heavy.$$
    run $PY scripts/micro_report.py "$OUT" --mode heavy ;;

  tune)
    run $PY scripts/tune.py ;;
  pilot)
    OUT=runs/pilot
    run $PY runner/bench_runner.py --outdir "$OUT" --max-seeds 2 --timeout-s 7200 --jobs "$JOBS"
    chain "$OUT" || true ;;
  triage)
    run $PY scripts/log_triage.py runs/pilot ;;

  full)
    run $PY scripts/micro_report.py runs/micro_heavy --mode gate || {
      echo "[full] REFUSED: micro-heavy has not cleared the RAM budget." | tee -a "$RAW"; exit 1; }
    OUT=runs/full
    run $PY runner/bench_runner.py --outdir "$OUT" --timeout-s 7200 --jobs "$JOBS"
    chain "$OUT" || true ;;
  final)
    OUT=runs/final_$(date -u +%Y%m%d)
    if [ -d "$OUT" ]; then echo "[final] $OUT exists -- refusing (a fresh dir is required)"; exit 1; fi
    run $PY runner/bench_runner.py --outdir "$OUT" --no-resume --timeout-s 7200 --jobs "$JOBS"
    if chain "$OUT"; then
      run $PY scripts/make_figures.py "$OUT"
      run $PY scripts/make_tables.py "$OUT"
      echo "[final] GATE PASSED on $OUT -- tables and figures regenerated; freezable" | tee -a "$RAW"
    else
      echo "[final] GATE FAILED on $OUT -- tables NOT regenerated, numbers NOT freezable" | tee -a "$RAW"
    fi ;;

  stats) chain "${OUT:?set OUT=runs/...}" ;;
  gate)  run $PY scripts/review_gate.py "${OUT:?set OUT=runs/...}" ;;
  *) echo "unknown stage: $STAGE"; exit 2 ;;
esac
