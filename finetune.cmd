executable = finetune_model.sh
getenv = true
error = finetune_model.error
log = finetune_model.log
output = finetune_model.out
notification = complete
arguments = ""
transfer_executable = false
request_memory = 16000
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
queue
