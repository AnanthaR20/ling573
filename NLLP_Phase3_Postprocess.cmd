############
# LED-base, 2048 input (-ATS, + binary blanks) using min [NO_SUMMARY] likelihood
############
executable = nllp_postprocess_evaluate.sh
getenv = true
environment = "CLUSTER_ID=$(Cluster); PROCESS_ID=$(Process);"
arguments = --config_id 5 --new_config_id 12 --base_tokenizer allenai/led-base-16384 \
--checkpoint /home2/jcmw614/ling573/models/led-base/billsum_clean_train_se3-led-2048-512/binary_blank_targets/2048_512_5_epochs/checkpoint-45220/ \
--k_selector rouge --k_limit 2 --batch_size 4
transfer_executable = false
output = postprocess_empties.$(Cluster).$(Process).out
error = postprocess_empties.$(Cluster).$(Process).err
log = postprocess_empties.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 6000
notification = error
queue

############
# LED-base, 1024 input (-ATS, + binary blanks) using min [NO_SUMMARY] likelihood
############
executable = nllp_postprocess_evaluate.sh
getenv = true
environment = "CLUSTER_ID=$(Cluster); PROCESS_ID=$(Process);"
arguments = --config_id 4 --new_config_id 13 --base_tokenizer allenai/led-base-16384 \
--checkpoint /home2/jcmw614/ling573/models/led-base/billsum_clean_train_se3-led-1024-512/binary_blank_targets/1024_512_5_epochs/checkpoint-108471/ \
--k_selector rouge --k_limit 3 --batch_size 4
transfer_executable = false
output = postprocess_empties.$(Cluster).$(Process).out
error = postprocess_empties.$(Cluster).$(Process).err
log = postprocess_empties.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 6000
notification = error
queue
