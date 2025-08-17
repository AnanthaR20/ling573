############
# Gold: Redundancy (sanity check)
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file eval/gold_reference_metrics.csv --metric redundancy --batch_size 8
transfer_executable = false
output = patch_metric.gold.redundancy.out
error = patch_metric.gold.redundancy.err
log = patch_metric.gold.redundancy.log
request_memory = 1000
notification = error
queue

############
# Config 0: Redundancy
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file output/0.led-base.billsum_clean_train_se3-led-1024-512.drop_blank_targets.1024_512_5_epochs.checkpoint-41832.csv \
--metric redundancy --batch_size 8
transfer_executable = false
output = patch_metric.0.redundancy.out
error = patch_metric.0.redundancy.err
log = patch_metric.0.redundancy.log
request_memory = 1000
notification = error
queue

############
# Config 1: Redundancy
############
executable = posthoc/patch_metric.sh
getenv = true
arguments = --file output/1.led-base.billsum_clean_train_se3-led-2048-512_simple.drop_blank_targets.2048_512_5_epochs.checkpoint-22804.csv \
--metric redundancy --batch_size 8
transfer_executable = false
output = patch_metric.1.redundancy.out
error = patch_metric.1.redundancy.err
log = patch_metric.1.redundancy.log
request_memory = 1000
notification = error
queue