cd /KungFu
ldconfig /usr/local/cuda-9.0/targets/x86_64-linux/lib/lib
env KUNGFU_USE_NCCL=1 pip3 install --no-index --user -U .
ldconfig
./scripts/go-install.sh