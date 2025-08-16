#!/bin/sh

# Log cluster and process ID
echo "Cluster $CLUSTER_ID Process $PROCESS_ID"
echo ""

# Source config file
CUDA_LAUNCH_BLOCKING=1 /home2/jcmw614/miniconda3/envs/573-env/bin/python posthoc/probe_control_token.py $@