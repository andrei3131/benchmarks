#!/usr/bin/env bash

NUM_WORKERS=4
NUM_EPOCHS=1
NOISE_FILES_PATH="/home/ab7515"
CHECKPOINTS_PREFIX="/data/kungfu"



train() {
    BATCH=$1
    TRAIN_ID=$2
    LR=$3
    # Checkpoint version ID
    VERSION_ID=$(printf "%06d" $(($2)))

    # 0 warm-up batches
    echo "[BEGIN TRAINING KEY] training-parallel-${TRAIN_ID}"
    kungfu-prun  -np ${NUM_WORKERS} -H 127.0.0.1:${NUM_WORKERS} -timeout 1000000s \
        python3 tf_cnn_benchmarks.py --model=resnet32 --data_name=cifar10 --data_dir=/data/cifar-10/cifar-10-batches-py \
        --num_epochs=50 \
        --num_gpus=1 \
        --eval=False \
        --forward_only=False \
        --num_warmup_batches=0 \
        --print_training_accuracy=True \
        --batch_size=${BATCH} \
        --momentum=0.9 \
        --weight_decay=0.001 \
        --optimizer=momentum \
        --staged_vars=False \
        --variable_update=kungfu \
        --kungfu_strategy=cpu_all_reduce_noise \
        --running_sum_interval=300 \
        --noise_decay_factor=0.01 \
        --future_batch_limit=512 \
        --use_datasets=True \
        --distortions=False \
        --fuse_decode_and_crop=True \
        --resize_method=bilinear \
        --display_every=1 \
        --run_version=$2 \
        --checkpoint_directory=${CHECKPOINTS_PREFIX}/train_dir \
        --checkpoint_every_n_epochs=True \
        --checkpoint_interval=0.25 \
        --data_format=NCHW \
        --batchnorm_persistent=True \
        --use_tf_layers=True \
        --winograd_nonfused=True
    echo "[END TRAINING KEY] training-parallel-${TRAIN_ID}"
}

validate() {
    VALIDATION_ID=$1

    # Checkpoint version ID
    VERSION_ID=$(printf "%06d" $(($1)))

    echo "VALIDATION VERSION_ID ${VERSION_ID}"
    for worker in 0 # 1 2 3 
    do
    echo "[BEGIN VALIDATION KEY] validation-parallel-worker-${worker}-validation-id-${VALIDATION_ID}"
    python3 tf_cnn_benchmarks.py --eval=True --forward_only=False --model=resnet32 --data_name=cifar10 \
        --data_dir=/data/cifar-10/cifar-10-batches-py \
        --variable_update=replicated --data_format=NCHW --use_datasets=False --num_epochs=1 --eval_batch_size=50 \
        --num_gpus=4 --use_tf_layers=True \
        --checkpoint_directory=${CHECKPOINTS_PREFIX}/train_dir/v-${VERSION_ID} \
        --checkpoint_interval=0.25 \
        --checkpoint_every_n_epochs=False 
    echo "[END VALIDATION KEY] validation-parallel-worker-${worker}-validation-id-${VALIDATION_ID}"
    done
}

NEW_NOISE_FILE_NAME="${NOISE_FILES_PATH}/median-noise.txt"


FUTURE_BATCH=64
i="1"
while [ $i -le $NUM_EPOCHS ]
do

    LR=""
    if [ $i -lt 120 ]
    then
        LR="0.01"    
    fi
    if [ $i -lt 80 ]
    then
        LR="0.1"    
    fi
    if [ $i -ge 120 ]
    then
        LR="0.001"    
    fi

    echo "EPOCH ${i}"
    echo "LEARNING RATE $LR"
    # Train
    start=`date +%s`
    train ${FUTURE_BATCH} ${i} ${LR}
    end=`date +%s`
    runtime_train=$((end-start))
    echo "Train ${i} took: ${runtime_train}"

    # Validate
    start=`date +%s`
    validate ${i}
    end=`date +%s`
    runtime_val=$((end-start))
    echo "Validation ${i} took: ${runtime_val}"

    # Restore
    i=$[$i+1]
done

