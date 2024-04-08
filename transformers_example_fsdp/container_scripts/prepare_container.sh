#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
source $SCRIPT_DIR/container_config.sh

# Change installed packages from requirements.txt 
install_command="pip install -r $SCRIPT_DIR/requirements.txt"
finish_string="Installation finished"
singularity exec $BINDS $CONTAINER \
	bash -c	"source /opt/miniconda3/bin/activate pytorch; \
	$install_command; \
	echo $finish_strings"
