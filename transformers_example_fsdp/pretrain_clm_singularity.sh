#!/bin/bash

#SBATCH --job-name=cl_v3
#SBATCH --nodes=4
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --partition=dev-g
#SBATCH --time=00-01:00:00
#SBATCH --gpus-per-node=mi250:8
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account=project_462000353
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --exclude=nid005003,nid007971,nid007972

export HF_HOME=/scratch/project_462000086/risto/transformers_cache

# distributed setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export PROCESSES=$((SLURM_GPUS_ON_NODE * SLURM_JOB_NUM_NODES))

ln -f -s "${SLURM_JOB_ID}.out" logs/latest.out
ln -f -s "${SLURM_JOB_ID}.err" logs/latest.err
mkdir -p logs

# compilers in the container
export CC=gcc-10
export CXX=g++-10

mkdir -p workdir
wd=$(realpath workdir)

export PYTHONUSERBASE=/scratch/project_462000086/risto/lumi-tools/container_scripts/pythonuserbase
CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.0.sif
SING_BIND="/scratch/project_462000319/,/scratch/project_462000086/,$PYTHONUSERBASE,$TRANSFORMERS_CACHE,$wd"

FSDP_TYPE="auto_wrap"
cat <<EOF >accelerate_config.yaml
    compute_environment: LOCAL_MACHINE
    debug: false
    distributed_type: FSDP
    downcast_bf16: 'no'
    fsdp_config:
        fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
        fsdp_backward_prefetch: BACKWARD_PRE
        fsdp_cpu_ram_efficient_loading: true
        fsdp_forward_prefetch: false
        fsdp_offload_params: false
        fsdp_sharding_strategy: FULL_SHARD
        fsdp_state_dict_type: SHARDED_STATE_DICT
        fsdp_sync_module_states: true
        fsdp_use_orig_params: true
    machine_rank: 0
    main_process_ip: $MASTER_ADDR
    main_process_port: 9999
    main_training_function: main
    mixed_precision: bf16
    num_machines: $SLURM_NNODES
    num_processes: $PROCESSES
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false
EOF


MODEL=/scratch/project_462000319/general-tools/checkpoints/33B_torch_step95328_bfloat16/
CMD=" \
    run_clm.py \
    --model_name_or_path $MODEL \
    --output_dir finetuned_poro_paraphrase \
    --train_file data/paraphrase_corpus.txt \
    --do_train \
    --logging_steps 10 \
    --per_device_train_batch_size 1  \
    --per_device_eval_batch_size 1 \
    --max_train_samples 100 \
    --bf16 \
    --bf16_full_eval \
    --block_size 1024 \
    --fsdp $FSDP_TYPE
    "

if [ ! -d "$wd"/cray-deps ] ; then
  rm -rf "$wd"/cray-deps
  mkdir "$wd"/cray-deps
  cp /usr/lib64/libcxi* $wd/cray-deps
fi

# OTHER LAUNCHERS CAN BE USED HERE
export LAUNCHER="accelerate launch \
    --config_file accelerate_config.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $PROCESSES \
    --num_machines $SLURM_NNODES \
    "

CMD="$LAUNCHER $CMD"


echo "$CMD"

srun \
    --label \
    singularity exec \
    -B /var/spool/slurmd \
    -B /opt/cray \
    -B /usr/lib64/libcxi.so.1 \
    -B /usr/lib64/libjansson.so.4 \
    -B "$SING_BIND" \
    -B "$PWD" \
    "$CONTAINER" \
    ./launch.sh \
    $CMD
