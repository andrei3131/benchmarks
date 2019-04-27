#!/usr/bin/env bash

set -x

cd /home/work/user-job-dir/tf_cnn_benchmarks/

mkdir /cache/checkpoints/

kungfu-prun -np $1 -H $2 -timeout 3600s \
    python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=150 \
    --model=resnet50_v1.5 --optimizer=momentum --variable_update=kungfu --kungfu_strategy=nccl_all_reduce \
    --num_gpus=1 --num_epochs=2 --num_warmup_batches=0 --weight_decay=1e-4 --print_training_accuracy=True \
    --single_l2_loss_op=True --loss_type_to_report=base_loss --nodistortions --data_name=imagenet \
    --checkpoint_interval=1 --checkpoint_directory=/cache/checkpoints/