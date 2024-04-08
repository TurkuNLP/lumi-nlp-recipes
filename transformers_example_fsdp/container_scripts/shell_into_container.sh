#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR=$(dirname "$0")

# Source the config file from the same directory
source "$SCRIPT_DIR/container_config.sh"

# Call singularity to execute a shell which sources our environment within the container and gives us an interactive shell instance
singularity exec $BINDS $CONTAINER bash -c "source /opt/miniconda3/bin/activate pytorch; bash"
