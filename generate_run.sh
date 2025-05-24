#!/bin/bash

# Check if config file was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

CONFIG_FILE=$1

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found"
    exit 1
fi

# Parse the config file
source $CONFIG_FILE

# Validate required parameters
if [ -z "$PLATFORM" ]; then
    echo "Error: PLATFORM not specified in config file"
    exit 1
fi

if [ -z "$CHECKPOINT" ]; then
    echo "Error: CHECKPOINT not specified in config file"
    exit 1
fi

if [ -z "$MODE" ]; then
    echo "Error: MODE not specified in config file"
    exit 1
fi

# Log mode parameter
if [ $MODE = "predict" ] ; then
    echo "Generating model predictions only"
else
    echo "Running end-to-end system"
fi

# Generate files based on PLATFORM
case "$PLATFORM" in
    patas)
        echo "Preparing to run on Patas..."
        # spacy model download
        /home2/jcmw614/miniconda3/envs/573-env/bin/python -m spacy download en_core_web_sm
        # Generate Condor .cmd file
        output_file="run_model.cmd"
        
        cat > "$output_file" <<EOF
executable = run_model.sh
getenv = true
arguments = --checkpoint $CHECKPOINT --mode $MODE --testfile $TESTFILE  --concat $CONCAT --batch_size $BATCH_SIZE
transfer_executable = false
output = run_model.\$(Cluster).out
error = run_model.\$(Cluster).err
log = run_model.\$(Cluster).log
request_GPUs = 1
request_memory = 3000
queue
EOF
        
        echo "Generated Condor submit file: $output_file"
        condor_submit $output_file
        ;;
        
        
    hyak)
        echo "Preparing to run on Hyak..."
        # spacy model download
        /gscratch/scrubbed/jcmw614/envs/573-env/bin/python -m spacy download en_core_web_sm
        # Generate Slurm .sbatch file
        output_file="run_model.slurm"
        
        cat > "$output_file" <<EOF
#!/bin/bash
#SBATCH --job-name=run_model
#SBATCH --output=out/%x/%j.out
#SBATCH --error=error/%x/%j.err
#SBATCH --log=log/%x/%j.log
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --mail-user=$USER@uw.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

apptainer exec --cleanenv --bind /gscratch /gscratch/scrubbed/jcmw614/project.sif /gscratch/scrubbed/jcmw614/envs/573-env/bin/python run_model.py --checkpoint $CHECKPOINT --mode $MODE --testfile $TESTFILE --concat $CONCAT --batch_size $BATCH_SIZE
EOF
        
        echo "Generated Slurm submit file: $output_file"
        sbatch $outputfile
        ;;
        
    *)
        echo "Error: Unknown PLATFORM '$PLATFORM'. Supported PLATFORMs are 'patas' and 'hyak'"
        exit 1
        ;;
esac