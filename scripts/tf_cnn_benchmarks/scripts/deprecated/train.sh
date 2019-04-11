#!/bin/bash
#
# Benchmark(s) for TensorFlow.
#

if [ $# -ne 3 ]; then
   echo "usage: train.sh [b] [g] [s]"
fi
my_b=$1
my_g=$2
my_s=$3

printf "Train batch size %4d on %d GPU devices\n" $my_b $my_g

FLAGS=

#
# Basic configuration
#
# Run benchmark in training mode
EVALUATION="False"

MODEL="resnet32"

DATASET="cifar10"
# Synthetic data
# DATADIR=""
# Binary data
# DATADIR="/data/tf/cifar-10/cifar-10-batches-bin/cifar-10-batches-py/"
# TF record data
DATADIR="/mnt/nfs/users/piwatcha/my-tensorflow/benchmarks/scripts/my-benchmarks/data/cifar-10"
# On platypus, the records exist under "/data/tf/cifar-10/cifar-10-batches-bin/"

# Only one of the two must be set
N_EPOCH=160
N_BATCH=0

FLAGS=${FLAGS}" --eval=${EVALUATION} --forward_only=False"
FLAGS=${FLAGS}" --model=${MODEL}"
FLAGS=${FLAGS}" --data_name=${DATASET}"
[ -n "${DATADIR}" ] && FLAGS=${FLAGS}" --data_dir=${DATADIR}"
FLAGS=${FLAGS}" --num_batches=${N_BATCH}"
FLAGS=${FLAGS}" --num_epochs=${N_EPOCH}"

# Defaults

# By default --job_name is '' for single-server execution
FLAGS=${FLAGS}" --task_index=0"
FLAGS=${FLAGS}" --print_training_accuracy=True" # Print training accuracy, why not...
# Set default random seed
FLAGS=${FLAGS}" --tf_random_seed=1234"

#
# CPU/GPU configuration
#
GPU_COUNT=$my_g
GPU_MODE="gpu_private"
GPU_THREAD_COUNT=2

FLAGS=${FLAGS}" --num_gpus=${GPU_COUNT}"
FLAGS=${FLAGS}" --gpu_thread_mode=${GPU_MODE}"
FLAGS=${FLAGS}" --per_gpu_thread_count=${GPU_THREAD_COUNT}"

# Defaults

# By default --gpu_indices is None
# By default --debugger is None"
FLAGS=${FLAGS}" --device=gpu"
FLAGS=${FLAGS}" --num_inter_threads=0"
FLAGS=${FLAGS}" --num_intra_threads=1"
FLAGS=${FLAGS}" --num_warmup_batches=0"
FLAGS=${FLAGS}" --gpu_memory_frac_for_testing=0.0"

#
# Hyper-parameters
#
BATCH_SIZE=$my_b
LEARNING_RATE=
LEARNING_RATE_SCHEDULE0="0.05;80;0.005;120;0.0005"
LEARNING_RATE_SCHEDULE1="0.1;80;0.01;120;0.001" # LR0;E1;LR1;...;En;LRn
LEARNING_RATE_SCHEDULE2="0.2;80;0.02;120;0.002" 
LEARNING_RATE_SCHEDULE3="0.3;80;0.03;120;0.003" 
LEARNING_RATE_SCHEDULE4="0.4;80;0.04;120;0.004" 
LEARNING_RATE_SCHEDULE5="0.5;80;0.05;120;0.005"

LEARNING_RATE_SCHEDULE=${my_s} # {LEARNING_RATE_SCHEDULE1}

MOMENTUM=0.9
WEIGHT_DECAY="0.0001"
# Configure optimiser
#
# Note: MSR has not installed bc, so assume momentum is set
# HAS_MOMENTUM=`echo "$MOMENTUM > 0.0" | bc -l`
HAS_MOMENTUM=1
if [ $HAS_MOMENTUM -eq 0 ]; then
    OPTIMISER="sgd"
else
    OPTIMISER="momentum"
fi

FLAGS=${FLAGS}" --batch_size=${BATCH_SIZE}"

# By default learning_rate is None, in which case we fall back to a model-specific learning rate
[ -n "${LEARNING_RATE}" ] && FLAGS=${FLAGS}" --learning_rate=${LEARNING_RATE}"

# By default --piecewise_learning_rate_schedule is None
[ -n "${LEARNING_RATE_SCHEDULE}" ] && FLAGS=${FLAGS}" --piecewise_learning_rate_schedule=${LEARNING_RATE_SCHEDULE}"

FLAGS=${FLAGS}" --momentum=${MOMENTUM}"
FLAGS=${FLAGS}" --weight_decay=${WEIGHT_DECAY}"
FLAGS=${FLAGS}" --optimizer=${OPTIMISER}"

# Defaults

# By default --gradient_clip is None
FLAGS=${FLAGS}" --num_epochs_per_decay=0.0"
FLAGS=${FLAGS}" --learning_rate_decay_factor=0.0"
FLAGS=${FLAGS}" --minimum_learning_rate=0.0"

# We never use RMSProp optimiser
FLAGS=${FLAGS}" --rmsprop_decay=0.9"
FLAGS=${FLAGS}" --rmsprop_epsilon=1.0"
FLAGS=${FLAGS}" --rmsprop_momentum=0.9"

#
# Synchronisation parameters
#
VAR_UPDATE="parameter_server" # parameter_server, replicated, independent
VAR_STAGED="False"
ALL_REDUCE="nccl" # empty, nccl, or xring

FLAGS=${FLAGS}" --variable_update=${VAR_UPDATE}"
FLAGS=${FLAGS}" --staged_vars=${VAR_STAGED}"
# By default specification is None, so don't set it unless required
[ -n "${ALL_REDUCE}" ] && FLAGS=${FLAGS}" --all_reduce_spec=${ALL_REDUCE}"

# Defaults

FLAGS=${FLAGS}" --local_parameter_device=gpu"
FLAGS=${FLAGS}" --agg_small_grads_max_bytes=0"
FLAGS=${FLAGS}" --agg_small_grads_max_group=10"
FLAGS=${FLAGS}" --hierarchical_copy=False"
FLAGS=${FLAGS}" --horovod_device=''"


#
# Image producer configuration
#
batch_group_size=2

FLAGS=${FLAGS}" --batch_group_size=${batch_group_size}"

# Defaults

FLAGS=${FLAGS}" --cache_data=False"
FLAGS=${FLAGS}" --use_datasets=True"
FLAGS=${FLAGS}" --use_python32_barrier=False"

#
# Image pre-processor configuration
#
# Defaults

FLAGS=${FLAGS}" --distort_color_in_yiq=True"
FLAGS=${FLAGS}" --distortions=True"
FLAGS=${FLAGS}" --fuse_decode_and_crop=True"
FLAGS=${FLAGS}" --resize_method=bilinear"

#
# Checkpoints, summaries, etc.
#
# Fix display interval to every 100 tasks...
DISPLAY_INTERVAL=100

CHECKPOINT_MANUALLY="True"
# We want 1 checkpoint per epoch... With 180 epochs in total, we will have 360 points
CHECKPOINT_INTERVAL=1
CHECKPOINT_DIRECTORY="$HOME/my-tensorflow/benchmarks/scripts/my-benchmarks/checkpoints"

FLAGS=${FLAGS}" --display_every=${DISPLAY_INTERVAL}"

FLAGS=${FLAGS}" --checkpoint_manually=${CHECKPOINT_MANUALLY}"
FLAGS=${FLAGS}" --checkpoint_interval=${CHECKPOINT_INTERVAL}"
FLAGS=${FLAGS}" --checkpoint_directory=${CHECKPOINT_DIRECTORY}"

# Defaults

FLAGS=${FLAGS}" --eval_dir=None"
FLAGS=${FLAGS}" --eval_interval_secs=0"
FLAGS=${FLAGS}" --save_model_secs=0"
FLAGS=${FLAGS}" --save_summaries_steps=0"
FLAGS=${FLAGS}" --summary_verbosity=0"
# By default --trace_file is ''
# By default --train_dir is None
# By default --graph_file is None
# By default --result_storage is None

#
# Model parameters
#
# Defaults

FLAGS=${FLAGS}" --data_format=NCHW"
FLAGS=${FLAGS}" --batchnorm_persistent=True"
FLAGS=${FLAGS}" --use_tf_layers=True"
FLAGS=${FLAGS}" --winograd_nonfused=True"

# Run.
#
python tf_cnn_benchmarks.py $FLAGS > results/v4/b-${my_b}-g-${my_g}.out 2>&1

# --autotune_threshold=None
# --controller_host=None
# --cross_replica_sync=True
# --enable_layout_optimizer=False
# --force_gpu_compatible=True
# --use_fp16=False
# --fp16_enable_auto_loss_scale=False
# --fp16_inc_loss_scale_every_n=1000
# --fp16_loss_scale=1
# --fp16_vars=False
# --kmp_affinity='granularity=fine,verbose,compact,1,0'
# --kmp_blocktime=30
# --kmp_settings=1
# --single_l2_loss_op=False
# --loss_type_to_report="total_loss"
# --mkl=False
# --worker_hosts=""
# --ps_hosts=""
# --sync_on_finish=False
# --server_protocol="grpc"
# --rewriter_config=None
# --use_chrome_trace_format="true"
# --xla=False
