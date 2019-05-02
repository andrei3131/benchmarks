#!/usr/bin/env bash

set -x

cd /home/work/user-job-dir/code/KungFu/

./configure --build-tensorflow-ops
ldconfig /usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs/
env KUNGFU_USE_NCCL=0 pip install --no-index --user -U . 
ldconfig
./scripts/go-install.sh

cd /home/work/user-job-dir/code/benchmarks/scripts/tf_cnn_benchmarks

python moxing/prepare_input.py

# Use 2>&1 | to redirect stderr to tee file

# KUNGFU_STRATEGY_1="partial_exchange"

# KUNGFU_STRATEGY_2="partial_exchange_accumulation"

# KUNGFU_STRATEGY_3="partial_exchange_accumulation_avg_peers"

# KUNGFU_STRATEGY_4="partial_exchange_accumulation_avg_window"

# 0.008 fails, 0.0085 fails
# 0.1 partial_exchage works

# for KUNGFU_STRATEGY in ${KUNGFU_STRATEGY_1} ${KUNGFU_STRATEGY_2} ${KUNGFU_STRATEGY_3} ${KUNGFU_STRATEGY_4}
# do
#     for fraction in 0.1
#     do
#         # Train model using Ako
#         echo "[BEGIN TRAINING KEY] training-${KUNGFU_STRATEGY}-fraction-${fraction}\n"
#         kungfu-prun  -np 16 -H 169.254.128.207:8,169.254.128.185:8 -nic ib0 -timeout 1000000s \
#             python tf_cnn_benchmarks.py --model=resnet50 --data_name=imagenet --data_dir=/cache/data_dir --train_dir=/cache/train_dir \
#             --num_epochs=1 --eval=False --forward_only=False --print_training_accuracy=True --tf_random_seed=123456789 \
#             --num_gpus=1 --gpu_thread_mode=gpu_private --num_warmup_batches=20 --batch_size=150 \
#             --piecewise_learning_rate_schedule="0.1;80;0.01;120;0.001" --momentum=0.9 --weight_decay=0.0001 \
#             --optimizer=momentum --variable_update=kungfu --staged_vars=False --kungfu_strategy=${KUNGFU_STRATEGY} \
#             --partial_exchange_fraction=${fraction} --use_datasets=True --distortions=False --fuse_decode_and_crop=True \
#             --resize_method=bilinear --display_every=100 --checkpoint_every_n_epochs=True --checkpoint_interval=1 \
#             --checkpoint_directory=/cache/checkpoints-${KUNGFU_STRATEGY}-fraction-${fraction} \
#             --data_format=NCHW --batchnorm_persistent=True --use_tf_layers=True --winograd_nonfused=True 
#         echo "[END TRAINING KEY] training-${KUNGFU_STRATEGY}-fraction-${fraction}\n"
#         # Evaluate the checkpoint and print the accuracy over epochs
#         # for worker in 0 1 2 3 4 # Use all 16!!
#         # do
#         # echo "[BEGIN VALIDATION KEY] validation-${KUNGFU_STRATEGY}-fraction-${fraction}-worker-${worker}\n"
#         # python tf_cnn_benchmarks.py --eval=True --forward_only=False --model=resnet50 --data_name=imagenet --data_dir=/cache/data_dir \
#         #     --variable_update=replicated --data_format=NCHW --use_datasets=False --num_epochs=1 --eval_batch_size=150 \
#         #     --tf_random_seed=123456789 --num_gpus=8 --use_tf_layers=True \
#         #     --checkpoint_directory=/cache/checkpoints-${KUNGFU_STRATEGY}-fraction-${fraction}-worker-${worker}/v-000001 --checkpoint_interval=1 --checkpoint_every_n_epochs=True 
#         # echo "[END VALIDATION KEY] validation-${KUNGFU_STRATEGY}-fraction-${fraction}-worker-${worker}\n"
#         #done
#     done
# done 


echo "[BEGIN TRAINING KEY] training-parallel-sgd\n"
kungfu-prun -np 4 -H 127.0.0.1:4 -timeout 1000000s # -np 16 -H 169.254.128.207:8,169.254.128.185:8 -nic ib0 -timeout 1000000s \
    python tf_cnn_benchmarks.py --model=resnet50 --data_name=imagenet --data_dir=/cache/data_dir --train_dir=/cache/train_dir \
    --num_batches=1000 --eval=False --forward_only=False --print_training_accuracy=True --tf_random_seed=123456789 \
    --num_gpus=1 --gpu_thread_mode=gpu_private --num_warmup_batches=20 --batch_size=150 \
    --piecewise_learning_rate_schedule="0.1;80;0.01;120;0.001" --momentum=0.9 --weight_decay=0.0001 \
    --optimizer=momentum --variable_update=kungfu --staged_vars=False --kungfu_strategy=cpu_all_reduce \
    --use_datasets=True --distortions=False --fuse_decode_and_crop=True \
    --resize_method=bilinear --display_every=100 --checkpoint_every_n_epochs=True --checkpoint_interval=1 \
    --checkpoint_directory=/cache/checkpoints-parallel \
    --data_format=NCHW --batchnorm_persistent=True --use_tf_layers=True --winograd_nonfused=True 
echo "[END TRAINING KEY] training-parallel-sgd\n"


python moxing/prepare_output.py