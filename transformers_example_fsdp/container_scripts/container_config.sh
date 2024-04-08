#!/bin/bash
export CONTAINER="/appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.0.sif"
CALLER_PATH=$(dirname "${BASH_SOURCE[1]}")

# relative to the caller script 
export PYTHONUSERBASE="$CALLER_PATH/pythonuserbase" 
mkdir -p $PYTHONUSERBASE

BIND_PATHS="/scratch/project_462000086,/scratch/project_462000319/,/scratch/project_462000444,/flash/project_462000319"
export BINDS="-B $PWD -B $BIND_PATHS"

echo "Set CONTAINER=$CONTAINER"
echo "PYTHONUSERBASE=$PYTHONUSERBASE"
echo "BINDS=$BINDS"
echo "SCRIPT_DIR $CALLER_PATH"d