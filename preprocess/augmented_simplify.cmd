executable = simplify.sh
getenv     = true
output     = reduced_simplify_se3.out
error      = reduced_simplify_se3.error
log        = reduced_simplify_se3.log
notification = complete
arguments = "--from_toy --split test --chunk_type se3-t5-512-512"
transfer_executable = false
request_memory = 3000
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
queue
