#!/usr/bin/env bash

set -x

cd /home/work/user-job-dir/tf_cnn_benchmarks/

mkdir /cache/checkpoints/

chmod +x moxing/kube-plm-rsh-agent

# echo "$BATCH_TASK_CURRENT_INSTANCE slots=$1" > moxing/hostfile
python mpi_hostfile.py -s 8 -q false -f ./moxing/

# python moxing/prepare_input.py

if [ "$DLS_TASK_INDEX" = "0" ]
then
    mpirun --allow-run-as-root -np $1 -H $2 \
        -mca plm_rsh_agent moxing/kube-plm-rsh-agent \
        --hostfile moxing/hostfile \
        python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=150 \
        --model=resnet50_v1.5 --optimizer=momentum --variable_update=horovod \
        --num_gpus=1 --num_epochs=2 --num_warmup_batches=0 --weight_decay=1e-4 --print_training_accuracy=True \
        --single_l2_loss_op=True --loss_type_to_report=base_loss --nodistortions --data_name=imagenet \
        --checkpoint_interval=1 --checkpoint_directory=/cache/checkpoints/ # --use_fp16
else
    sleep 5d
fi

# python moxing/prepare_output.py

