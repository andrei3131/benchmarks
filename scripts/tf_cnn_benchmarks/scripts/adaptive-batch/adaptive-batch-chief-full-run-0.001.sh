#!/usr/bin/env bash

NUM_WORKERS=4
NUM_EPOCHS=140
NOISE_FILES_PATH="/home/ab7515"
CHECKPOINTS_PREFIX="/data/kungfu"

train() {
    BATCH=$1
    TRAIN_ID=$2

    # Checkpoint version ID
    VERSION_ID=$(printf "%06d" $(($2)))

    # 0 warm-up batches
    echo "[BEGIN TRAINING KEY] training-parallel-${TRAIN_ID}"
    kungfu-prun  -np ${NUM_WORKERS} -H 127.0.0.1:${NUM_WORKERS} -timeout 1000000s \
        python3 tf_cnn_benchmarks.py --model=resnet32 --data_name=cifar10 --data_dir=/data/cifar-10/cifar-10-batches-py \
        --num_epochs=1 \
        --num_gpus=1 \
        --eval=False \
        --forward_only=False \
        --num_warmup_batches=20 \
        --print_training_accuracy=True \
        --batch_size=${BATCH} \
        --momentum=0.9 \
        --weight_decay=0.001 \
        --optimizer=momentum \
        --staged_vars=False \
        --variable_update=kungfu \
        --kungfu_strategy=cpu_all_reduce_noise \
        --running_sum_interval=300 \
        --noise_decay_factor=0.001 \
        --use_datasets=True \
        --distortions=False \
        --fuse_decode_and_crop=True \
        --resize_method=bilinear \
        --display_every=1 \
        --run_version=$2 \
        --train_dir=${CHECKPOINTS_PREFIX}/train_dir/v-${VERSION_ID} \
        --checkpoint_directory=${CHECKPOINTS_PREFIX}/train_dir \
        --checkpoint_every_n_epochs=True \
        --checkpoint_interval=1 \
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
        --variable_update=replicated --data_format=NCHW --use_datasets=False --num_batches=50 --eval_batch_size=150 \
        --num_gpus=4 --use_tf_layers=True \
        --checkpoint_directory=${CHECKPOINTS_PREFIX}/train_dir/v-${VERSION_ID} \
        --checkpoint_every_n_epochs=False 
    echo "[END VALIDATION KEY] validation-parallel-worker-${worker}-validation-id-${VALIDATION_ID}"
    done
}

compute_median() {
    FILE=$1
    sort -n ${FILE} | awk 'NF{a[NR]=$1;c++}END{print (c%2==0)?((a[c/2]+a[c/2+1])/2):a[c/2+1]}'
}

get_median() {
    nums=$(<$1); 
    list=(`for n in $nums; do printf "%10.06f\n" $n; done | sort -n`); 
    #echo min ${list[0]}; 
    #echo max ${list[${#list[*]}-1]}; 
    echo ${list[${#list[*]}/2]};
}


get_future_batch() {
    NEW_NOISE_FILE_NAME=$1
    for worker in {0..3}
    do
        FILE_NAME="${NOISE_FILES_PATH}/noise-worker-${worker}.txt"
        FUTURE_BATCH=$(grep  -Eo '^[-+]?[0-9]+\.?[0-9]*$' ${FILE_NAME} | tail -100)
        echo ${FUTURE_BATCH} >> ${NEW_NOISE_FILE_NAME}
    done

    FUTURE_BATCH=$(get_median ${NEW_NOISE_FILE_NAME})
    FUTURE_BATCH=$(printf '%.0f\n' ${FUTURE_BATCH})
    echo "${FUTURE_BATCH}"
    rm -f ${NEW_NOISE_FILE_NAME}
}


NEW_NOISE_FILE_NAME="${NOISE_FILES_PATH}/median-noise.txt"


FUTURE_BATCH=32
i="1"
while [ $i -le $NUM_EPOCHS ]
do
    # Train
    start=`date +%s`
    train ${FUTURE_BATCH} ${i}
    end=`date +%s`
    runtime_train=$((end-start))
    echo "Train ${i} took: ${runtime_train}"

    # Compute future batch
    start=`date +%s`
    FUTURE_BATCH=$(get_future_batch ${NEW_NOISE_FILE_NAME})
    end=`date +%s`
    runtime_batch_change=$((end-start))
    echo "[${runtime_batch_change} seconds] FUTURE_BATCH is ${FUTURE_BATCH}"

    # Validate
    start=`date +%s`
    validate ${i}
    end=`date +%s`
    runtime_val=$((end-start))
    echo "Validation ${i} took: ${runtime_val}"

    # Restore
    i=$[$i+1]
done

