# Starter example of training Poro with FSDP using HuggingFace 

## Environment
I'm using singularity-containers which have PyTorch, Apex, aws-ofi-rccl-plugin for libfabric and flash-attn for rocm as my base python environment. As singularity-containers are immutable,
additional packages get installed with pip --user, and by setting PYTHONUSERPATH-environment variable we can set this userspace location to an arbitrary location. 
`container_scripts/prepare_container.sh` sets up a container and installs packages from `container_scripts/requirements.txt`

### Setting up
```git clone https://github.com/luukkonenr/lumi-tools.git
cd lumi-tools
bash container_scripts/prepare_container.sh
```

### Opening a shell within the container
```
bash container_scripts/shell_into_container.sh
```

## Running FSDP example multi-node on LUMI

