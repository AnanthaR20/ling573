############
# Config 0: SummaC
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file output/0.led-base.billsum_clean_train_se3-led-1024-512.drop_blank_targets.1024_512_5_epochs.checkpoint-41832.csv \
--metric summac 
transfer_executable = false
output = patch_metric.0.summac.out
error = patch_metric.0.summac.err
log = patch_metric.0.summac.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 5000
notification = error
queue

############
# Config 12: SummaC
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file output/12.led-base.billsum_clean_train_se3-led-2048-512.binary_blank_targets.2048_512_5_epochs.checkpoint-45220.csv \
--metric summac 
transfer_executable = false
output = patch_metric.12.summac.out
error = patch_metric.12.summac.err
log = patch_metric.12.summac.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 5000
notification = error
queue

############
# Config 13: SummaC
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file output/13.led-base.billsum_clean_train_se3-led-1024-512.binary_blank_targets.1024_512_5_epochs.checkpoint-108471.csv \
--metric summac 
transfer_executable = false
output = patch_metric.13.summac.out
error = patch_metric.13.summac.err
log = patch_metric.13.summac.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 5000
notification = error
queue

############
# Baseline LED-base: SummaC
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file output/deliverable_4/led-base/led-base_billsum_clean_test_se3-led-2048-512.csv \
--metric summac 
transfer_executable = false
output = patch_metric.13.summac.out
error = patch_metric.13.summac.err
log = patch_metric.13.summac.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 5000
notification = error
queue

############
# Config 1: SummaC
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file output/1.led-base.billsum_clean_train_se3-led-2048-512_simple.drop_blank_targets.2048_512_5_epochs.checkpoint-22804.csv \ 
--metric summac 
transfer_executable = false
output = patch_metric.1.summac.out
error = patch_metric.1.summac.err
log = patch_metric.1.summac.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 5000
notification = error
queue