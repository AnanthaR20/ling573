#!/bin/bash

#SBATCH --job-name=simplify_job
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=15G
#SBATCH --time=12:00:00
#SBATCH -o log/%j_%x.out
#SBATCH --error=error/%j_%x.error
#SBATCH --mail-user=$USER@uw.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# command:
apptainer exec --cleanenv --bind /gscratch ../../project.sif /gscratch/scrubbed/jcmw614/envs/573-env/bin/python simplify.py --split test --chunk_type se3 --checkpoint t5