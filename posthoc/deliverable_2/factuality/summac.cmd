executable = summac_eval.sh
getenv     = true
output     = condor_logs/summac_eval.out
error      = condor_logs/summac_eval.error
log        = condor_logs/summac_eval.log
notification = complete
arguments = ""
transfer_executable = false
request_memory = 2*1024
queue
