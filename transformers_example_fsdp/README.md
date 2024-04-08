# Starter example of training Poro with FSDP using HuggingFace 

Disclaimer: This is a very light and un-optimized example. This is just merely to show a one way to launch training with FSDP. For a proper, working solution, please use sft_trl.
Remember to change the hardcoded paths.

## Environment
I'm using singularity-containers which have PyTorch, Apex, aws-ofi-rccl-plugin for libfabric and flash-attn for rocm as my base python environment. As singularity-containers are immutable,
additional packages get installed with pip --user, and by setting PYTHONUSERPATH-environment variable we can set this userspace location to an arbitrary location. 
`container_scripts/prepare_container.sh` sets up a container and installs packages from `container_scripts/requirements.txt`

### Setting up
```git clone https://github.com/luukkonenr/lumi-tools.git
cd lumi-tools
bash container_scripts/prepare_container.sh
```

### Prepare data

```
bash container_scripts/shell_into_container.sh
python3 prepare_paraphrase_corpus.py
```

### Running the training

Edit the launcher `pretrain_clm_singularity.sh` to point to your directories, model you want to train etc. 

`sbatch pretrain_clm_singularity.sh`



### Simplest example:

Simply training with english wikipedia
https://github.com/luukkonenr/lumi-tools/tree/main/finetuning/transformers_example
