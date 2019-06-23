#!/usr/bin/env bash

set -x

cd /home/work/user-job-dir/benchmarks-fresh/scripts/tf_cnn_benchmarks/

python moxing/prepare_input.py


echo "[BEGIN TRAINING KEY] training-parallel"
kungfu-prun -np 8 -H 127.0.0.1:8 -timeout 1000000s \
    python tf_cnn_benchmarks.py --model=resnet50 \
    --data_name=imagenet \
    --data_dir=/cache/data_dir \
    --num_epochs=90 \
    --eval=False \
    --forward_only=False \
    --print_training_accuracy=True \
    --num_gpus=1 \
    --num_warmup_batches=20 \
    --batch_size=256 \
    --momentum=0.9 \
    --weight_decay=0.0001 \
    --staged_vars=False \
    --optimizer=p2p_averaging \
    --variable_update=kungfu \
    --kungfu_strategy=none \
    --model_averaging_device=gpu \
    --request_mode=async \
    --peer_selection_strategy=roundrobin \
    --use_datasets=True \
    --distortions=False \
    --fuse_decode_and_crop=True \
    --resize_method=bilinear \
    --display_every=100 \
    --checkpoint_every_n_epochs=True \
    --checkpoint_interval=0.125 \
    --checkpoint_directory=/cache/checkpoints-parallel \
    --data_format=NCHW \
    --batchnorm_persistent=True \
    --use_tf_layers=True \
    --winograd_nonfused=True 
echo "[END TRAINING KEY] training-parallel"


for worker in 0 1 2 3 4 5 6 7
do
echo "[BEGIN VALIDATION KEY] validation-parallel-worker-${worker}"
python tf_cnn_benchmarks.py --eval=True --forward_only=False --model=resnet50 --data_name=imagenet --data_dir=/cache/data_dir \
    --variable_update=replicated --data_format=NCHW --use_datasets=False --num_batches=50 --eval_batch_size=150 \
    --num_gpus=8 --use_tf_layers=True \
    --checkpoint_directory=/cache/checkpoints-parallel-worker-${worker}/v-000001 --checkpoint_interval=0.125 --checkpoint_every_n_epochs=True 
echo "[END VALIDATION KEY] validation-parallel-worker-${worker}"
done


#python moxing/prepare_output.py