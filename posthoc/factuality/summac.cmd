executable = summac_eval.sh
getenv     = true
output     = summac_eval.out
error      =summac_eval.error
log        = summac_eval.log
notification = complete
arguments = ""
transfer_executable = false
request_memory = 16000
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
queue
