#!/bin/bash
#SBATCH --job-name=env_setup 
#SBATCH --partition=dev-g  
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1     
#SBATCH --gpus-per-node=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=7
#SBATCH --time=00:10:00
#SBATCH --account=project_462000558
#SBATCH -o logs/%x.out
#SBATCH -e logs/%x.err

mkdir -p logs

# Load modules
module load LUMI #Loads correct compilers for the accelerators, propably not needed
module use /appl/local/csc/modulefiles/ #Add the module path needed for csc modules in Lumi
module load pytorch/2.1 #The latest pytorch module has issues with venv as of 25.3.2024


#Create venv
python -m venv .venv --system-site-packages

#Activate
source .venv/bin/activate

# Install pip packages
python -m pip install peft
python -m pip install trl