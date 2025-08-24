############
# Se3-2048: Alignscore
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file eval/se3-2048_baseline_metrics.csv --metric alignscore --batch_size 8
transfer_executable = false
output = patch_metric.se3-2048.alignscore.out
error = patch_metric.se3-2048.alignscore.err
log = patch_metric.se3-2048.alignscore.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 4000
notification = error
queue

############
# Se3-1024: Alignscore
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file eval/se3-1024_baseline_metrics.csv --metric alignscore --batch_size 8
transfer_executable = false
output = patch_metric.se3-1024.alignscore.out
error = patch_metric.se3-1024.alignscore.err
log = patch_metric.se3-1024.alignscore.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 4000
notification = error
queue

############
# Se3-2048: LFTK
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file eval/se3-2048_baseline_metrics.csv --metric lftk
transfer_executable = false
output = patch_metric.se3-2048.lftk.out
error = patch_metric.se3-2048.lftk.err
log = patch_metric.se3-2048.lftk.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 4000
notification = error
queue

############
# Se3-1024: LFTK
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file eval/se3-1024_baseline_metrics.csv --metric lftk
transfer_executable = false
output = patch_metric.se3-1024.lftk.out
error = patch_metric.se3-1024.lftk.err
log = patch_metric.se3-1024.lftk.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 4000
notification = error
queue

############
# Se3-2048: SummaC
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file eval/se3-2048_baseline_metrics.csv --metric summac
transfer_executable = false
output = patch_metric.se3-2048.summac.out
error = patch_metric.se3-2048.summac.err
log = patch_metric.se3-2048.summac.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 4000
notification = error
queue

############
# Se3-1024: SummaC
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file eval/se3-1024_baseline_metrics.csv --metric summac
transfer_executable = false
output = patch_metric.se3-1024.summac.out
error = patch_metric.se3-1024.summac.err
log = patch_metric.se3-1024.summac.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 4000
notification = error
queue