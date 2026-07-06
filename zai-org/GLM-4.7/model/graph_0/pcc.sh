#!/bin/bash
# PCC check: run the decode forward with golden comparison. Run inside docker.
cd /home/mvasiljev/tt-xla && source venv/activate
G0=/home/mvasiljev/tt-metal/ttnn-models/zai-org/GLM-4.7/model/graph_0
cd "$G0"
export TT_MLIR_HOME PYTHONPATH
RT="$TT_MLIR_HOME/third_party/tt-metal/src/tt-metal"
export TT_METAL_RUNTIME_ROOT="$RT"
export PYTHONPATH="$RT/ttnn:$RT/tools/:$PYTHONPATH"
GLM_CHECK_PCC=1 python3 main.py > "$G0/pcc_full.log" 2>&1
echo "=== exit $? ; filtered tail (full log: pcc_full.log) ==="
grep -iE "PCC|floor|freed|Error|assert|Traceback|hang|deadlock|incompatible|>>> |RuntimeError|FAILED" "$G0/pcc_full.log" | tail -50
