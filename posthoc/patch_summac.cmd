############
# Gold: SummaC
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file eval/gold_reference_metrics.csv --metric summac --batch_size 8
transfer_executable = false
output = patch_metric.gold.summac.out
error = patch_metric.gold.summac.err
log = patch_metric.gold.summac.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 5000
notification = error
queue

############
# Config 5: SummaC
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file eval/gold_reference_metrics.csv --metric summac --batch_size 8
transfer_executable = false
output = patch_metric.5.summac.out
error = patch_metric.5.summac.err
log = patch_metric.5.summac.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 5000
notification = error
queue

############
# Config 4: SummaC
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file eval/gold_reference_metrics.csv --metric summac --batch_size 8
transfer_executable = false
output = patch_metric.4.summac.out
error = patch_metric.4.summac.err
log = patch_metric.4.summac.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 5000
notification = error
queue