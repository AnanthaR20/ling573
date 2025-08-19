#!/bin/sh

# Log cluster and process ID
echo "Cluster $CLUSTER_ID Process $PROCESS_ID"
echo ""

# Source config file
/home2/jcmw614/miniconda3/envs/573-env/bin/python nllp_postprocess_evaluate.py $@