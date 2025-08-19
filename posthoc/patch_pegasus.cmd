############
# Deliverable 2: Redundancy
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file output/deliverable_2/pegasusbillsum_baseline_ALL_metrics.csv \
--metric redundancy --batch_size 8
transfer_executable = false
output = patch_metric.pegasus.redundancy.out
error = patch_metric.pegasus.redundancy.err
log = patch_metric.pegasus.redundancy.log
request_memory = 1000
notification = error
queue

############
# Deliverable 2: BERTScore
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file output/deliverable_2/pegasusbillsum_baseline_ALL_metrics.csv \
--metric bertscore --batch_size 8
transfer_executable = false
output = patch_metric.pegasus.bertscore.out
error = patch_metric.pegasus.bertscore.err
log = patch_metric.pegasus.bertscore.log
request_memory = 4000
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
notification = error
queue

############
# Deliverable 2: Alignscore
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file output/deliverable_2/pegasusbillsum_baseline_ALL_metrics.csv \
--metric alignscore --batch_size 8
transfer_executable = false
output = patch_metric.pegasus.alignscore.out
error = patch_metric.pegasus.alignscore.err
log = patch_metric.pegasus.alignscore.log
request_memory = 4000
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
notification = error
queue