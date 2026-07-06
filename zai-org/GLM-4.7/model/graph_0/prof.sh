#!/bin/bash
# Profiling + reporting helper. Run inside docker container with tt-xla venv.
# Usage: ./prof.sh            -> tracy run + per-signpost reports into profiles_latest/
#        ./prof.sh report_only <csv>  -> just regenerate reports from an existing csv
set -e
cd /home/mvasiljev/tt-xla && source venv/activate
RT="$TT_MLIR_HOME/third_party/tt-metal/src/tt-metal"
G0=/home/mvasiljev/tt-metal/ttnn-models/zai-org/GLM-4.7/model/graph_0
cd "$G0"

if [[ "$1" == "report_only" ]]; then
    CSV="$2"
else
    echo ">>> tracy run..."
    ./run -t 2>&1 | tail -5
    CSV=$(find "$G0/generated/profiler/reports" "$RT/generated/profiler/reports" -name "ops_perf_results*.csv" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)
fi
echo ">>> CSV=$CSV"

OUT="$G0/profiles_latest"
mkdir -p "$OUT"
PR=/home/mvasiljev/.local/bin/tt-perf-report
# signpost pairs: (start end label)
gen() { $PR "$CSV" --start-signpost "$1" --end-signpost "$2" > "$OUT/$3.txt" 2>/dev/null || true; }
gen preamble  L0_attn  preamble
gen L3_attn   L3_mlp   L3_attn
gen L2_attn   L2_mlp   L2_attn
gen L2_mlp    L3_attn  L2_mlp
gen L3_mlp    lm_head  L3_mlp
gen lm_head   ""       lm_head
echo ">>> reports written to $OUT"
echo "$CSV" > "$OUT/.csvpath"
