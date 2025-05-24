# ling573
Repository for CLMS LING 573 group project. Evaluated on the [BillSum corpus](https://huggingface.co/datasets/FiscalNote/billsum), taking in a plaintext legislative bill document and generating a summary of its contents.

# Conda first-time setup
1. Clone repository if it does not already exist
```cmd
git clone git@github.com:AnanthaR20/ling573.git
```
2. Download miniconda [for your OS](https://www.anaconda.com/docs/getting-started/miniconda/main)

3. Create a new environment:
```cmd
conda create -n 573-env
```
4. Activate the environment to start developing! Yay!
```cmd
conda activate 573-env
```
5. Install all the required packages:
```cmd
pip install -r requirements.txt
```
6. Download the optimized spaCy English language model for evaluation 
```cmd
python -m spacy download en_core_web_sm
```

Note that Conda environments have a slightly different setup if you run the wugwATSS
system on Hyak. Official documentation can be found [here](https://hyak.uw.edu/docs/tools/python/)

# Virtual environment usage
Virtual environment invocation is managed by our controller shell 
script, `generate_run.sh` by specifying the correct config file,
`patas.config` or `hyak.config`. To explain each parameter provided in
the config:

```bash
# Specify the platform that is running the augmented system; the only valid options are "patas" or "hyak"
PLATFORM="patas" 
# any Huggingface Seq2Seq model for summarization can be used here
CHECKPOINT="google/pegasus-billsum" 
# For Deliverable 3, we only need to generate model predictions on the preprocessed data
MODE="predict" 
# Filepath relative to the root folder of this repository
TESTFILE="preprocess/data/billsum_clean_test_se3-t5-512-512.csv" 
# For Deliverable 3, we only need to reconstruct full summaries after summarizing chunks
CONCAT="post"
# Patas and Hyak can handle different batch sizes due to memory constraints
BATCH_SIZE=4
```
# Augmented system
Note that the test data is directly preloaded in the `preprocess` directory of this repository so no extra data downloads are necessary to run the system. Additionally, all ROUGE and readability formulas are evaluated at test time and handled by the controller script at the last stage.

## Patas (NVIDIA Quadro 8000)
1. SSH into Patas
```cmd
ssh <UW NetID>@patas.ling.washington.edu
cd ling573
conda activate 573-env
```
### Clean
```cmd
python preprocess/clean.py
```
This script takes approximately 1-2 minutes, and can be run directly on the Patas head node.
### Chunk
To create the semantic self-segmented chunks from BillSum documents, we directly call the Se3 submodule:
```cmd
cd preprocess/se3/
git submodule init
git submodule update
git pull
condor_submit augmented_segment.cmd
```
This stage takes approximately 2-3 hours to run as a Condor job on the entire test split. This also requires running Se3's metric learning script, which we describe briefly in the ATS section.

### Simplify
To simplify the document segments, we navigate back to the preprocess module:
```cmd
cd ..
condor_submit augmented_simplify.cmd
```
Alternatively, this can be run on Hyak:
```cmd
cd ..
sbatch 
```
On Hyak, this step takes approximately 2 hours to run.

### Generate summaries

Run the controller script which will generate and submit a Condor job on your behalf:
```cmd
cd ling573
./generate_run.sh patas.config
# Attend for successful Condor job submission message
```
If running on Hyak, the controller script will generate a SLURM batch job, build an Apptainer with the Conda environment and submit the job on your behalf:
```cmd
cd /gscratch/scrubbed/jcmw614/ling573
./generate_run.sh hyak.config
# Attend for successful SLURM job submission message
```
Note: this takes approximately 40 min for the first 15 documents of the test set, which were segmented into 130 text chunks.

# Baseline system
Note that test data is directly loaded with the `datasets` library so no extra arguments are needed on the command line.
## Patas (NVIDIA Quadro 8000)
1. SSH into Patas
```cmd
ssh <UW NetID>@patas.ling.washington.edu
```
2. Submit condor job
```cmd
condor_submit run_baseline/run_baseline.cmd
```
2. Wait...
3. Find your output in `run_baseline.out`

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

After testing that a pre-existing ATS code repo behaves as expected when tested independently, we add the repository in the `preprocess` child directory.

## Se3
A prerequisite to run the wugwATSs system is to have access to adjusted Legal-BERT sentence embeddings as instantiated by the Se3 repository. This can be handled by a Condor job in approximately 7 hours:
```cmd
cd preprocess/se3
condor_submit learning.cmd

