#!/bin/bash

#SBATCH --job-name=simplify_job
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10G
#SBATCH --time=12:00:00
#SBATCH -o log/%j_%x.out

# command:
apptainer exec --cleanenv --bind /gscratch ../../project.sif /gscratch/scrubbed/jcmw614/envs/573-env/bin/python simplify.py --split test --toy 1 --chunk_type se3 --checkpoint t5