## Build

The following command builds a custom image for running tensorflow-benchmarks and Horovod.

```bash
$ docker build -f hvd-benchmark.Dockerfile -t hvd-bench:latest .
```