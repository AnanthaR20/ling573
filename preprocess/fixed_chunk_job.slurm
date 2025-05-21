#!/bin/bash

#SBATCH --job-name=chunktrain_fixed_job
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=5G
#SBATCH --time=8:00:00
#SBATCH -o log/%x_%j.out

# command:
apptainer exec --cleanenv --bind /gscratch ../../project.sif /gscratch/scrubbed/jcmw614/envs/573-env/bin/python chunk.py