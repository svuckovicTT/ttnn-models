#!/bin/bash
# Tracy profiling helper. Run INSIDE docker (tt-xla venv). Writes profiler
# artifacts to container-LOCAL /tmp (overlay disk) instead of the NFS cwd: the
# tracy report step cp's the ~1.4GB profile_log_device.csv, which HANGS on the
# 98%-full NFS. Local /tmp makes that cp fast. Only the small (~17MB)
# ops_perf_results CSV is copied back to the NFS graph_0 dir so the host-side
# tt-perf-report (report.sh) can read it. Prints the NFS CSV path at the end.
set -e
cd /home/mvasiljev/tt-xla && source venv/activate
RT="$TT_MLIR_HOME/third_party/tt-metal/src/tt-metal"
export TT_METAL_RUNTIME_ROOT="$RT"
export PYTHONPATH="$RT/ttnn:$RT/tools/:$PYTHONPATH"
G0=/home/mvasiljev/tt-metal/ttnn-models/zai-org/GLM-4.7/model/graph_0
LOCAL=/tmp/glm_prof
rm -rf "$LOCAL"; mkdir -p "$LOCAL"
cd "$G0"

echo ">>> tracy run (output -> local $LOCAL) ..."
python3 -m tracy -r -m -v -p -o "$LOCAL" \
    --tracy-tools-folder "$RT/build/tools/profiler/bin/" main 2>&1 | tail -6

# find the ops_perf CSV among local (preferred) and any fallback locations
CSV=$(find "$LOCAL" "$G0/generated/profiler/reports" "$RT/generated/profiler/reports" \
    -name "ops_perf_results*.csv" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)
echo ">>> local CSV=$CSV"
# copy the small CSV back to NFS for host-side tt-perf-report
DEST="$G0/generated/profiler/reports_local"
mkdir -p "$DEST"
NFSCSV="$DEST/$(basename "$CSV")"
cp "$CSV" "$NFSCSV"
echo ">>> NFS CSV=$NFSCSV"
# kill any straggler giant-file cp the tracy tooling may have spawned onto NFS
pkill -KILL -u "$(id -u)" -f "cp .*profile_log_device.csv" 2>/dev/null || true
