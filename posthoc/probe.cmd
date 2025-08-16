############
# LED-base, 2048 input (+ATS, +blanks)
############
executable = posthoc/probe.sh
getenv = true
environment = "CLUSTER_ID=$(Cluster); PROCESS_ID=$(Process);"
arguments = --config_id 4 --base_tokenizer allenai/led-base-16384 \
--checkpoint /home2/jcmw614/ling573/models/led-base/billsum_clean_train_se3-led-1024-512/binary_blank_targets/1024_512_5_epochs/checkpoint-108471/ \
--mode test \
--testfile preprocess/nllp_data/billsum_clean_test_se3-led-1024-512.csv \
--batch_size 6 
transfer_executable = false
output = probe_binary_phase_2.$(Cluster).$(Process).out
error = probe_binary_phase_2.$(Cluster).$(Process).err
log = probe_binary_phase_2.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 5400
notification = error
queue

############
# LED-base, 1024 input (-ATS, +blanks)
############
