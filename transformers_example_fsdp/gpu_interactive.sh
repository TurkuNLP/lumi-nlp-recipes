#!/bin/bash
srun \
    --account=project_462000353 \
    --partition=dev-g \
    --ntasks=1 \
    --gres=gpu:mi250:8 \
    --time=01:00:00 \
    --mem=0 \
    --pty \
    bash
