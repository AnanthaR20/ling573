############
# Gold: Alignscore
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file eval/gold_reference_metrics.csv --metric alignscore --batch_size 8
transfer_executable = false
output = patch_metric.gold.alignscore.out
error = patch_metric.gold.alignscore.err
log = patch_metric.gold.alignscore.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 4000
notification = error
queue

############
# Config 0: BERTScore
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file output/0.led-base.billsum_clean_train_se3-led-1024-512.drop_blank_targets.1024_512_5_epochs.checkpoint-41832.csv \
--metric bertscore --batch_size 8
transfer_executable = false
output = patch_metric.0.bertscore.out
error = patch_metric.0.bertscore.err
log = patch_metric.0.bertscore.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
notification = error
queue

############
# Config 1: BERTScore
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file output/1.led-base.billsum_clean_train_se3-led-2048-512_simple.drop_blank_targets.2048_512_5_epochs.checkpoint-22804.csv \
--metric bertscore --batch_size 8
transfer_executable = false
output = patch_metric.1.bertscore.out
error = patch_metric.1.bertscore.err
log = patch_metric.1.bertscore.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
notification = error
queue

