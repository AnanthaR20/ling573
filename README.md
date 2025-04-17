# ling573
Repo for CLMS LING 573 group project

(TBD: Diagram for how this will work)

# Conda first-time setup

1. Download miniconda [for your OS](https://www.anaconda.com/docs/getting-started/miniconda/main)

2. Create a new environment:
```cmd
conda create -n 573-env
```
3. Activate the environment to start developing! Yay!
```cmd
conda activate 573-env
```
4. Install all the required packages:
```cmd
pip install -r requirements.txt
```

# Development
1. Activate conda environment:
```cmd
conda activate 573-env
```

# ATS development

After testing that a pre-existing ATS code repo works independently, we mirror that repository in the `preprocess` child directory and only make edits as needed to create that mirrored repo's corresponding virtual environment.

## (WIP) ATS virtual env 1
instructions coming soon!
## (WIP) ATS virtual env 2
instructions coming soon!

In our final finetuning + testing processes, we use `conda` to manage virtual environment switching between subprocesses (data preprocessing, model tuning, inference/evaluation).
