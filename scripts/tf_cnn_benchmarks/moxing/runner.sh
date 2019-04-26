#!/usr/bin/env bash

set -x

cd /home/work/user-job-dir/tf_cnn_benchmarks/

chmod +x moxing/kube-plm-rsh-agent
echo "$BATCH_TASK_CURRENT_INSTANCE slots=$1" > moxing/hostfile
cat moxing/hostfile

python moxing/prepare_input.py

mpirun --allow-run-as-root -np $1 -H $2 \
    -mca plm_rsh_agent /home/work/user-job-dir/tf_cnn_benchmarks/moxing/kube-plm-rsh-agent \
    --hostfile /home/work/user-job-dir/tf_cnn_benchmarks/moxing/hostfile \
    --display-map \
    python tf_cnn_benchmarks.py $3

python moxing/prepare_output.py