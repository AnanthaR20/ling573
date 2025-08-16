############
# LED-base, 2048 input (+ATS, +blanks)
############
executable = nllp_evaluate_model.sh
getenv = true
environment = "CLUSTER_ID=$(Cluster); PROCESS_ID=$(Process);"
arguments = --config_id 4 --base_tokenizer allenai/led-base-16384 \
--checkpoint /home2/jcmw614/ling573/models/led-base/billsum_clean_train_se3-led-2048-512_simple/binary_blank_targets/2048_512_5_epochs/checkpoint-33528/ \
--mode test \
--testfile preprocess/nllp_data/billsum_clean_test_se3-led-2048-512.csv \
--batch_size 6 
transfer_executable = false
output = evaluate_phase_2.$(Cluster).$(Process).out
error = evaluate_phase_2.$(Cluster).$(Process).err
log = evaluate_phase_2.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 6000
notification = error
queue
