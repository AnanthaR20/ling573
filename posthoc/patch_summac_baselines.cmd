############
# Deliverable 4: AlignScore
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file output/deliverable_4/led-base/led-base_billsum_clean_test_se3-led-2048-512.csv \
--metric alignscore --batch_size 8 
transfer_executable = false
output = patch_metric.d4.alignscore.out
error = patch_metric.d4.alignscore.err
log = patch_metric.d4.alignscore.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 5000
notification = error
queue

############
# Deliverable 2: SummaC
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file output/deliverable_2/pegasusbillsum_baseline_ALL_metrics.csv \
--metric summac 
transfer_executable = false
output = patch_metric.pegasus.summac.out
error = patch_metric.pegasus.summac.err
log = patch_metric.pegasus.summac.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 5000
notification = error
queue

############
# Deliverable 4: SummaC
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file output/deliverable_4/led-base/led-base_billsum_clean_test_se3-led-2048-512.csv \
--metric summac 
transfer_executable = false
output = patch_metric.d4.summac.out
error = patch_metric.d4.summac.err
log = patch_metric.d4.summac.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 5000
notification = error
queue