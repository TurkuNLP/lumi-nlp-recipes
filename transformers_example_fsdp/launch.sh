#!/bin/bash

# Launch script used by slurm scripts, don't invoke directly.

source /opt/miniconda3/bin/activate pytorch

# Hoping to resolve "Cassini Event Queue overflow detected." errors
export FI_CXI_DEFAULT_CQ_SIZE=262144    # default 131072

echo "Rank $SLURM_PROCID CPU affinity: $(taskset -p $$)"

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export OMP_NUM_THREADS=1

export TORCH_EXTENSIONS_DIR=torch_extensions
mkdir -p $TORCH_EXTENSIONS_DIR

# debugging (noisy)
# export NCCL_DEBUG=WARN
# export RCCL_KERNEL_COLL_TRACE_ENABLE=1 
# export NCCL_DEBUG_SUBSYS=INIT,COLL
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HSA_ENABLE_INTERRUPT=0

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

echo "Launching on $SLURMD_NODENAME ($SLURM_PROCID/$SLURM_JOB_NUM_NODES)," \
     "master $MASTER_ADDR port $MASTER_PORT," \
     "GPUs $SLURM_GPUS_ON_NODE," \
     "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

export PATH=$PATH:$PYTHONUSERBASE/bin
eval "$@"
