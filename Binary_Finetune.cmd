############
# LED-base, 2048 input (-ATS)
############
executable = nllp_finetune_model.sh
getenv = true
environment = "CLUSTER_ID=$(Cluster); PROCESS_ID=$(Process);"
arguments = --checkpoint allenai/led-base-16384 --mode train \
--trainfile preprocess/nllp_data/billsum_clean_train_se3-led-2048-512.csv \
--batch_size 4 --blank_targets binary
transfer_executable = false
output = binary_phase_2.$(Cluster).$(Process).out
error = binary_phase_2.$(Cluster).$(Process).err
log = binary_phase_2.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 5000
notification = error
queue

############
# LED-base, 1024 input (-ATS)
############
executable = nllp_finetune_model.sh
getenv = true
environment = "CLUSTER_ID=$(Cluster); PROCESS_ID=$(Process);"
arguments = --checkpoint allenai/led-base-16384 --mode train \
--trainfile preprocess/nllp_data/billsum_clean_train_se3-led-1024-512.csv \
--batch_size 4 --blank_targets binary
transfer_executable = false
output = binary_phase_2.$(Cluster).$(Process).out
error = binary_phase_2.$(Cluster).$(Process).err
log = binary_phase_2.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 6000
notification = error
queue