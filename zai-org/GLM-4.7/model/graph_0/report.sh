#!/bin/bash
# HOST-side report generator: given an ops_perf CSV, produce per-signpost
# tt-perf-report txts into profiles_latest/. Run on host (python3.10 + tt_perf_report).
set -e
G0=/data/mvasiljev/tt-metal/ttnn-models/zai-org/GLM-4.7/model/graph_0
CSV="$1"
[[ -z "$CSV" ]] && CSV=$(find "$G0/generated/profiler/reports" -name "ops_perf_results*.csv" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2)
OUT="$G0/profiles_latest"; mkdir -p "$OUT"
PR=$(command -v tt-perf-report)
gen() { $PR "$CSV" --start-signpost "$1" ${2:+--end-signpost "$2"} > "$OUT/$3.txt" 2>/dev/null || true; }
gen preamble  L0_attn  preamble
gen L3_attn   L3_mlp   L3_attn
gen L2_attn   L2_mlp   L2_attn
gen L2_mlp    L3_attn  L2_mlp
gen L3_mlp    lm_head  L3_mlp
gen lm_head   ""       lm_head
echo "$CSV" > "$OUT/.csvpath"
echo ">>> reports from $CSV -> $OUT"
# print per-segment device-FW totals
for f in preamble L2_attn L3_attn L2_mlp L3_mlp lm_head; do
  t=$(grep -oE "^Total .*[0-9,]+ μs|signposts[[:space:]]+[0-9,.]+ μs|Device FW Duration[^0-9]*[0-9,.]+" "$OUT/$f.txt" 2>/dev/null | head -1)
  echo "  $f: ${t:-(see file)}"
done
