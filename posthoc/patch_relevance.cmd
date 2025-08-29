############
# Se3-2048: ROUGE
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file eval/se3-2048_baseline_metrics.csv --metric rouge --batch_size 8
transfer_executable = false
output = patch_metric.se3-2048.rouge.out
error = patch_metric.se3-2048.rouge.err
log = patch_metric.se3-2048.rouge.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 4000
notification = error
queue

############
# Se3-1024: ROUGE
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file eval/se3-1024_baseline_metrics.csv --metric rouge --batch_size 8
transfer_executable = false
output = patch_metric.se3-1024.rouge.out
error = patch_metric.se3-1024.rouge.err
log = patch_metric.se3-1024.rouge.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 4000
notification = error
queue

############
# Se3-2048: BERTScore
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file eval/se3-2048_baseline_metrics.csv --metric bertscore --batch_size 8
transfer_executable = false
output = patch_metric.se3-2048.bertscore.out
error = patch_metric.se3-2048.bertscore.err
log = patch_metric.se3-2048.bertscore.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 4000
notification = error
queue

############
# Se3-1024: BERTScore
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file eval/se3-1024_baseline_metrics.csv --metric bertscore --batch_size 8
transfer_executable = false
output = patch_metric.se3-1024.bertscore.out
error = patch_metric.se3-1024.bertscore.err
log = patch_metric.se3-1024.bertscore.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 4000
notification = error
queue