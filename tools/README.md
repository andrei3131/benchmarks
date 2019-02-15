## Build

The following command builds a custom image for running tensorflow-benchmarks and Horovod.

```bash
$ docker build -f hvd-benchmark.Dockerfile -t hvd-bench:latest .
```

## Run

The following command run tensorflow-benchmarks for ResNet50 using 4 local GPUs and synchronise through Horovod. Check more about [Horovod Docker](https://github.com/horovod/horovod/blob/master/docs/docker.md)

```bash
$ nvidia-docker run -it hvd-bench:latest
root@243d81c298a9:/benchmarks/scripts/tf_cnn_benchmarks# mpirun -np 4 -H localhost:4 python tf_cnn_benchmarks.py --num_gpus=1 --batch_size=32 --model=resnet50 --variable_update=horovod
```

If you don't run your container in privileged mode, you may see the following message:

```bash
[a8c9914754d2:00040] Read -1, expected 131072, errno = 1
```

You can ignore this message.