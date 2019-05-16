#!/usr/bin/env bash

set -x

cd /home/work/user-job-dir/tf_cnn_benchmarks/

# Use this when placing code in andrei/code
# cd /home/work/user-job-dir/code/kungfu/
# ./configure --build-tensorflow-ops
# ldconfig /usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs/
# env KUNGFU_USE_NCCL=0 pip install --no-index -U . 
# ldconfig
# ./scripts/go-install.sh
# cd /home/work/user-job-dir/code/benchmarks/scripts/tf_cnn_benchmarks

#python moxing/prepare_input.py

# Use 2>&1 | to redirect stderr to tee file


# Random seed has been removed from experiments

############ Single Machine
# echo "[BEGIN TRAINING KEY] training-parallel-sgd-fp16-local\n"
# kungfu-prun -np 8 -H 127.0.0.1:8 -timeout 1000000s \
#     python tf_cnn_benchmarks.py --use_fp16 --model=resnet50 --data_name=imagenet --data_dir=/cache/data_dir --train_dir=/cache/train_dir \
#     --num_epochs=2 --eval=False --forward_only=False --print_training_accuracy=True \
#     --num_gpus=1 --num_warmup_batches=20 --batch_size=310 \
#     --piecewise_learning_rate_schedule="0.1;80;0.01;120;0.001" --momentum=0.9 --weight_decay=0.0001 \
#     --optimizer=momentum --variable_update=kungfu --staged_vars=False --kungfu_strategy=cpu_all_reduce \
#     --use_datasets=True --distortions=False --fuse_decode_and_crop=True \
#     --resize_method=bilinear --display_every=100 --checkpoint_every_n_epochs=True --checkpoint_interval=1 \
#     --checkpoint_directory=/cache/checkpoints-parallel \
#     --data_format=NCHW --batchnorm_persistent=True --use_tf_layers=True --winograd_nonfused=True 
# echo "[END TRAINING KEY] training-parallel-sgd-fp16-local\n"

# rm -rf /cache/train_dir

# echo "[BEGIN TRAINING KEY] training-parallel-sgd-fp32-local\n"
# kungfu-prun -np  8 -H 127.0.0.1:8 -timeout 1000000s \
#     python tf_cnn_benchmarks.py --model=resnet50 --data_name=imagenet --data_dir=/cache/data_dir --train_dir=/cache/train_dir \
#     --num_epochs=2 --eval=False --forward_only=False --print_training_accuracy=True \
#     --num_gpus=1 --num_warmup_batches=20 --batch_size=150 \
#     --piecewise_learning_rate_schedule="0.1;80;0.01;120;0.001" --momentum=0.9 --weight_decay=0.0001 \
#     --optimizer=momentum --variable_update=kungfu --staged_vars=False --kungfu_strategy=cpu_all_reduce \
#     --use_datasets=True --distortions=False --fuse_decode_and_crop=True \
#     --resize_method=bilinear --display_every=100 --checkpoint_every_n_epochs=True --checkpoint_interval=1 \
#     --checkpoint_directory=/cache/checkpoints-parallel \
#     --data_format=NCHW --batchnorm_persistent=True --use_tf_layers=True --winograd_nonfused=True 
# echo "[END TRAINING KEY] training-parallel-sgd-fp32-local\n"

# rm -rf /cache/train_dir


############# Multi-Machine
# echo "[BEGIN TRAINING KEY] training-parallel-sgd-fp16\n"
# kungfu-prun -np 16 -H 169.254.128.207:8,169.254.128.185:8 -nic ib0 -timeout 1000000s \
#     python tf_cnn_benchmarks.py --use_fp16 --model=resnet50 --data_name=imagenet --data_dir=/cache/data_dir --train_dir=/cache/train_dir \
#     --num_epochs=1 --eval=False --forward_only=False --print_training_accuracy=Tru
#     --num_gpus=1 --num_warmup_batches=20 --batch_size=310 \
#     --piecewise_learning_rate_schedule="0.1;80;0.01;120;0.001" --momentum=0.9 --weight_decay=0.0001 \
#     --optimizer=momentum --variable_update=kungfu --staged_vars=False --kungfu_strategy=cpu_all_reduce \
#     --use_datasets=True --distortions=False --fuse_decode_and_crop=True \
#     --resize_method=bilinear --display_every=100 --checkpoint_every_n_epochs=True --checkpoint_interval=1 \
#     --checkpoint_directory=/cache/checkpoints-parallel \
#     --data_format=NCHW --batchnorm_persistent=True --use_tf_layers=True --winograd_nonfused=True 
# echo "[END TRAINING KEY] training-parallel-sgd-fp16\n"


# echo "[BEGIN TRAINING KEY] training-parallel-sgd-fp32\n"
# kungfu-prun -np 16 -H 169.254.128.207:8,169.254.128.185:8 -nic ib0 -timeout 1000000s \
#     python tf_cnn_benchmarks.py --model=resnet50 --data_name=imagenet --data_dir=/cache/data_dir --train_dir=/cache/train_dir \
#     --num_epochs=1 --eval=False --forward_only=False --print_training_accuracy=Tru
#     --num_gpus=1 --num_warmup_batches=20 --batch_size=150 \
#     --piecewise_learning_rate_schedule="0.1;80;0.01;120;0.001" --momentum=0.9 --weight_decay=0.0001 \
#     --optimizer=momentum --variable_update=kungfu --staged_vars=False --kungfu_strategy=cpu_all_reduce \
#     --use_datasets=True --distortions=False --fuse_decode_and_crop=True \
#     --resize_method=bilinear --display_every=100 --checkpoint_every_n_epochs=True --checkpoint_interval=1 \
#     --checkpoint_directory=/cache/checkpoints-parallel \
#     --data_format=NCHW --batchnorm_persistent=True --use_tf_layers=True --winograd_nonfused=True 
# echo "[END TRAINING KEY] training-parallel-sgd-fp32\n"



# KUNGFU_STRATEGY_1="partial_exchange"

# KUNGFU_STRATEGY_2="partial_exchange_accumulation"

# KUNGFU_STRATEGY_3="partial_exchange_accumulation_avg_peers"

# KUNGFU_STRATEGY_4="partial_exchange_accumulation_avg_window"

# # 0.008 fails, 0.0085 fails
# # 0.1 partial_exchage works

# rm -rf /cache/train_dir/*

#mkdir /cache/andrei-checkpoints-parallel

#mkdir /cache/andrei-checkpoints-ako-64


# for KUNGFU_STRATEGY in ${KUNGFU_STRATEGY_1} # ${KUNGFU_STRATEGY_2} ${KUNGFU_STRATEGY_3} ${KUNGFU_STRATEGY_4}
# do
#     for fraction in 0.1
#     do
#         # Train model using Ako
#         # Use default LR schedule: epochs 30 - 60 -80
#         echo "[BEGIN TRAINING KEY] training-fp32-${KUNGFU_STRATEGY}-fraction-${fraction}"
#         kungfu-prun  -np 8 -H 127.0.0.1:8 -timeout 1000000s \
#             python tf_cnn_benchmarks.py --model=resnet50 --data_name=imagenet --data_dir=/cache/data_dir --train_dir=/cache/train_dir \
#             --num_epochs=90 --eval=False --forward_only=False --print_training_accuracy=True \
#             --num_gpus=1 --num_warmup_batches=20 --batch_size=64 \
#             --momentum=0.9 --weight_decay=0.0001 \
#             --optimizer=momentum --variable_update=kungfu --staged_vars=False --kungfu_strategy=${KUNGFU_STRATEGY} \
#             --partial_exchange_fraction=${fraction} --use_datasets=True --distortions=False --fuse_decode_and_crop=True \
#             --resize_method=bilinear --display_every=100 --checkpoint_every_n_epochs=True --checkpoint_interval=0.125 \
#             --checkpoint_directory=/cache/checkpoints-${KUNGFU_STRATEGY}-fraction-${fraction} \
#             --data_format=NCHW --batchnorm_persistent=True --use_tf_layers=True --winograd_nonfused=True 
#         echo "[END TRAINING KEY] training-fp32-${KUNGFU_STRATEGY}-fraction-${fraction}"

#         cp -r /cache/checkpoints-partial_exchange-fraction-0.1-worker-* /cache/andrei-checkpoints-ako-64/

#         # Evaluate the checkpoint and print the accuracy over epochs
#         for worker in 0 1 2 3 4 5 6 7  
#         do
#         echo "[BEGIN VALIDATION KEY] validation-${KUNGFU_STRATEGY}-fraction-${fraction}-worker-${worker}"
#         python tf_cnn_benchmarks.py --eval=True --forward_only=False --model=resnet50 --data_name=imagenet --data_dir=/cache/data_dir \
#             --variable_update=replicated --data_format=NCHW --use_datasets=False --num_batches=50 --eval_batch_size=150 \
#             --num_gpus=8 --use_tf_layers=True \
#             --checkpoint_directory=/cache/checkpoints-${KUNGFU_STRATEGY}-fraction-${fraction}-worker-${worker}/v-000001 --checkpoint_interval=0.125 --checkpoint_every_n_epochs=True 
#         echo "[END VALIDATION KEY] validation-${KUNGFU_STRATEGY}-fraction-${fraction}-worker-${worker}"
#         done
#     done
# done 

# rm -rf /cache/checkpoints-partial_exchange-fraction-0.1-worker-*

#rm -rf /cache/train_dir/*

# echo "[BEGIN TRAINING KEY] training-parallel"
# kungfu-prun  -np 8 -H 127.0.0.1:8 -timeout 1000000s \
#     python tf_cnn_benchmarks.py --model=resnet50 --data_name=imagenet --data_dir=/cache/data_dir --train_dir=/cache/train_dir \
#     --num_epochs=90 --eval=False --forward_only=False --print_training_accuracy=True \
#     --num_gpus=1 --num_warmup_batches=20 --batch_size=64 \
#     --momentum=0.9 --weight_decay=0.0001 \
#     --optimizer=momentum --variable_update=kungfu --staged_vars=False --kungfu_strategy=cpu_all_reduce \
#     --use_datasets=True --distortions=False --fuse_decode_and_crop=True \
#     --resize_method=bilinear --display_every=100 --checkpoint_every_n_epochs=True --checkpoint_interval=0.125 \
#     --checkpoint_directory=/cache/checkpoints-parallel \
#     --data_format=NCHW --batchnorm_persistent=True --use_tf_layers=True --winograd_nonfused=True 
# echo "[END TRAINING KEY] training-parallel"

# for worker in 0 1 2 3 4 5 6 7  
# do
# echo "[BEGIN VALIDATION KEY] validation-parallel-worker-${worker}"
# python tf_cnn_benchmarks.py --eval=True --forward_only=False --model=resnet50 --data_name=imagenet --data_dir=/cache/data_dir \
#     --variable_update=replicated --data_format=NCHW --use_datasets=False --num_batches=50 --eval_batch_size=150 \
#     --num_gpus=8 --use_tf_layers=True \
#     --checkpoint_directory=/cache/checkpoints-parallel-worker-${worker}/v-000001 --checkpoint_interval=0.125 --checkpoint_every_n_epochs=True 
# echo "[END VALIDATION KEY] validation-parallel-worker-${worker}"
# done

# cp -r /cache/checkpoints-parallel-worker-* /cache/andrei-checkpoints-parallel/
# cp -r /cache/andrei-checkpoints-parallel /cache/train_dir/


# python tf_cnn_benchmarks.py \
#         --data_format=NCHW \
#         --batch_size=32 \
#         --model=resnet101 \
#         --optimizer=momentum \
#         --data_name=imagenet \
#         --variable_update=replicated \
#         --nodistortions \
#         --gradient_repacking=0 \
#         --num_gpus=8 \
#         --num_epochs=90 \
#         --weight_decay=1e-4 \
#         --data_dir=/cache/data_dir \
#         --checkpoint_interval=1 \
#         --checkpoint_directory=/cache/checkpoints-replicated \
#         --print_training_accuracy=True

nvidia-smi nvlink -s

#cp -r /cache/andrei-checkpoints-ako-64 /cache/train_dir/


# echo "[BEGIN TRAINING KEY] training-replicated-fp16\n" 
#     # Never run this with kungfu-prun
#     python tf_cnn_benchmarks.py --use_fp16 --model=resnet50 --data_name=imagenet --data_dir=/cache/data_dir --train_dir=/cache/train_dir \
#     --num_epochs=2 --eval=False --forward_only=False --print_training_accuracy=True --all_reduce_spec=nccl  #####!! CAREFULLLL \
#     --num_gpus=8 --num_warmup_batches=20 --batch_size=310 \
#     --piecewise_learning_rate_schedule="0.1;80;0.01;120;0.001" --momentum=0.9 --weight_decay=0.0001 \
#     --optimizer=momentum --variable_update=replicated --staged_vars=False \
#     --use_datasets=True --distortions=False --fuse_decode_and_crop=True \
#     --resize_method=bilinear --display_every=100 --checkpoint_every_n_epochs=True --checkpoint_interval=1 \
#     --checkpoint_directory=/cache/checkpoints-parallel \
#     --data_format=NCHW --batchnorm_persistent=True --use_tf_layers=True --winograd_nonfused=True 
# echo "[END TRAINING KEY] training-replicated-fp16\n"
#num_batches
# rm -rf /cache/train_dir

# # Batch sizes for fp16 and fp32 should never be the same
# echo "[BEGIN TRAINING KEY] training-replicated"
#     # Never run this with kungfu-prun
#     python tf_cnn_benchmarks.py --model=resnet50 --data_name=imagenet --data_dir=/cache/data_dir --train_dir=/cache/train_dir \
#     --num_epochs=50 --eval=False --forward_only=False --print_training_accuracy=True \
#     --num_gpus=8 --num_warmup_batches=20 --batch_size=64 \
#     --momentum=0.9 --weight_decay=0.0001 \
#     --optimizer=momentum --variable_update=replicated --staged_vars=False \
#     --use_datasets=True --distortions=False --fuse_decode_and_crop=True \
#     --resize_method=bilinear --display_every=100 --checkpoint_every_n_epochs=True --checkpoint_interval=1 \
#     --checkpoint_directory=/cache/checkpoints-replicated \
#     --data_format=NCHW --batchnorm_persistent=True --use_tf_layers=True --winograd_nonfused=True 
# echo "[END TRAINING KEY] training-replicated"

# echo "[BEGIN TRAINING KEY] validaton-replicated"
# python tf_cnn_benchmarks.py --eval=True --forward_only=False --model=resnet50 --data_name=imagenet --data_dir=/cache/data_dir \
#      --variable_update=replicated --data_format=NCHW --use_datasets=False --eval_batch_size=150 \
#      --num_gpus=8 --use_tf_layers=True \
#      --checkpoint_directory=/cache/checkpoints-replicated/v-000001 --checkpoint_interval=1 --checkpoint_every_n_epochs=True 
# echo "[END TRAINING KEY] validation-replicated"


#python moxing/prepare_output.py