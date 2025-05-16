# ling573
Repository for CLMS LING 573 group project. Evaluated on the [BillSum corpus](https://huggingface.co/datasets/FiscalNote/billsum), taking in a plaintext legislative bill document and generating a summary of its contents.

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

# Virtual environment usage
1. Activate conda environment:
```cmd
conda activate 573-env
```

# Baseline system
Note that test data is directly loaded with the `datasets` library so no extra arguments are needed on the command line.
## Patas (NVIDIA Quadro 8000)
1. SSH into Patas
```cmd
ssh <UW NetID>@patas.ling.washington.edu
```
2. Clone repository if it does not already exist
```cmd
git clone git@github.com:AnanthaR20/ling573.git
```
3. Submit condor job
```cmd
condor_submit run_baseline/run_baseline.cmd
```
4. Wait...
5. Find your output in `run_baseline.out`

## M1 Apple Silicon
1. Activate virtual environment (see above)
2. Run system from terminal
```cmd
python backup_run.py
```
3. Wait...but hopefully not as long!
4. In our ad-hoc backup run, output was directly printed to console and manually copied into a text file, `baseline_console.txt`. This console output can be aligned with the provided `title` column from BillSum using the `align()` function defined in `backup_run.py` and written to a CSV, `baseline_test.csv`. This CSV can be used for baseline evaluation.

## Baseline evaluation
1. Extract confidence intervals on ROUGE scores with `eval_metrics.py`
```cmd
cd eval
python eval_metrics.py
```
2. Extract readability scores in `eval_readability.ipynb` as we are still testing out different eval resources
```cmd
jupyter notebook
```
3. Use the provided readability scores to evaluate t-tests on each readability score

# ATS development

After testing that a pre-existing ATS code repo behaves as expected when tested independently, we plan to mirror that repository in the `preprocess` child directory.

## (WIP) Discourse Simplification
instructions coming soon!
## (WIP) T5: Split and Rephrase
instructions coming soon!

In our final finetuning + testing processes, we plan to use shell scripts `finetune_model.sh` and `run_model.sh` to manage the the intermediary data processing, inference, and evaluation steps. 

# Improved System 
## Data processing
...
## System Fine-tuning
...
## System Deployment
...
## System Evaluation
