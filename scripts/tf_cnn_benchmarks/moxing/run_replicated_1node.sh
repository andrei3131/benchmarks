#!/usr/bin/env bash

set -x

cd /home/work/user-job-dir/benchmarks-fresh/scripts/tf_cnn_benchmarks/

python moxing/prepare_input.py

echo "[BEGIN TRAINING KEY] training-parallel"
#  --kungfu_strategy=p2p_averaging \
#     --model_averaging_device=gpu \
#     --request_mode=async \

# kungfu-prun -np 8 -H 127.0.0.1:8 -timeout 1000000s \
#TONIGHT
python tf_cnn_benchmarks.py --model=resnet50 \
    --data_name=imagenet \
    --data_dir=/cache/data_dir \
    --num_epochs=30 \
    --eval=False \
    --forward_only=False \
    --print_training_accuracy=True \
    --num_gpus=8 \
    --num_warmup_batches=20 \
    --batch_size=128 \
    --momentum=0.9 \
    --weight_decay=0.0001 \
    --staged_vars=False \
    --optimizer=momentum \
    --variable_update=replicated \
    --use_datasets=True \
    --distortions=False \
    --fuse_decode_and_crop=True \
    --resize_method=bilinear \
    --display_every=100 \
    --checkpoint_every_n_epochs=True \
    --checkpoint_interval=1 \
    --checkpoint_directory=/cache/checkpoints-parallel \
    --data_format=NCHW \
    --batchnorm_persistent=True \
    --use_tf_layers=True \
    --winograd_nonfused=True 
echo "[END TRAINING KEY] training-parallel"



echo "[BEGIN VALIDATION KEY] validation-parallel-worker"
python tf_cnn_benchmarks.py --eval=True --forward_only=False --model=resnet50 --data_name=imagenet --data_dir=/cache/data_dir \
    --variable_update=replicated --data_format=NCHW --use_datasets=False --num_batches=50 --eval_batch_size=150 \
    --num_gpus=8 --use_tf_layers=True \
    --checkpoint_directory=/cache/checkpoints-parallel/v-000001 --checkpoint_interval=1 --checkpoint_every_n_epochs=True 
echo "[END VALIDATION KEY] validation-parallel-worker"



#python moxing/prepare_output.py