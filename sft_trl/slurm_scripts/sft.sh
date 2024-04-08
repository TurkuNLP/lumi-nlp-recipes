#!/bin/bash
#SBATCH --job-name=sft_poro
#SBATCH --account=project_462000558
#SBATCH --partition=dev-g
#SBATCH --cpus-per-task=56
#SBATCH --nodes=3
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

mkdir -p logs
rm -f logs/latest.out logs/latest.err
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.out logs/latest.out
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.err logs/latest.err

# auto-fail on any errors in this script
set -eo pipefail

# logging script's variables/commands for future debug needs
set -x

echo "PWD" $PWD

module use /appl/local/csc/modulefiles/ #It is adviced to add this into your .bashrc/.profile
module load pytorch/2.1 #The latest pytorch module seems to have issues with checkpoint saving/retrieving on multi-node
                        #https://github.com/huggingface/transformers/issues/27925 seems to be related

#Activate python venv
source /path/to/.venv/bin/activate

#Replace this to the venv you created
export PYTHONPATH="path/to/venv/lib/python3.10/site-packages"

export HF_HOME=/scratch/project_462000558/cache

#Variables for distributed enviroment
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))

#LOGGING/DEBUGGING
#export TORCH_DISTRIBUTED_DEBUG=DETAIL #Detailed stack trackes from worker failures
#export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
#export NCCL_DEBUG=INFO
#export HIP_LAUNCH_BLOCKING=1
#export HSA_FORCE_FINE_GRAIN_PCIE=1 #Supposedly improves performance/prevents hanging
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false #Removes error involved with the FastTokenizer and rust/python parallelism.
                                    #See more:https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996

#Accelerate config for distributed training
ACCELERATE_CONFIG_FILE=configs/accelerate/deepspeed_zero3.yaml
#Arguments for training
CONFIG_FILE=configs/poro/sft_full.yaml

echo "JOBNAME" $SLURM_JOB_NAME
echo "CONFIG" $CONFIG_FILE

export CMD=" \
    training/sft.py $CONFIG_FILE
    "


#LAUNCHER
export ACC_LAUNCHER="singularity_wrapper exec accelerate launch \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --config_file $ACCELERATE_CONFIG_FILE \
    --num_machines $SLURM_NNODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --role \$(hostname -s) \
    --tee 3 \
    "


SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$ACC_LAUNCHER --role \$SLURMD_NODENAME: $CMD"

echo "END TIME: $(date)"

echo "END $SLURM_JOBID: $(date)"
