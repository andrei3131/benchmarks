# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Benchmark script for TensorFlow.

See the README for more information.
"""

from __future__ import print_function

from absl import app
from absl import flags as absl_flags
import tensorflow as tf

import benchmark_cnn_kungfu
import cnn_util
import flags
from cnn_util import log_fn

# Import Huawei modelarts utils
import moxing as mox
import time
import os


flags.define_flags()
for name in flags.param_specs.keys():
  absl_flags.declare_key_flag(name)


def main(positional_arguments):
  # Command-line arguments like '--distortions False' are equivalent to
  # '--distortions=True False', where False is a positional argument. To prevent
  # this from silently running with distortions, we do not allow positional
  # arguments. 
  assert len(positional_arguments) >= 1
  if len(positional_arguments) > 1:
    raise ValueError('Received unknown positional arguments: %s'
                     % positional_arguments[1:])

  # START
  data_url = os.environ['DLS_DATA_URL']
  print('INFO: data_url: ' + data_url)
  train_url = os.environ['DLS_TRAIN_URL']
  print('INFO: train_url: ' + train_url)
  absl_flags.FLAGS.data_dir = '/cache/data_dir'
  absl_flags.FLAGS.train_dir = '/cache/train_dir'
  absl_flags.FLAGS.checkpoint_directory = absl_flags.FLAGS.train_dir

  n_tasks = int(os.environ['DLS_TASK_NUMBER'])
  print('INFO: n_tasks: ' + str(n_tasks))
  if n_tasks > 1:
    task_index = int(os.environ['DLS_TASK_INDEX'])
    print('INFO: task_index: ' + str(task_index))
    for i in range(0, n_tasks):
      host = os.environ['BATCH_CUSTOM' + str(i) +'_HOSTS']
      if i == 0:
        hosts = host
      else:
        hosts = hosts + ',' + host
    print('INFO: hosts: ' + hosts)

    absl_flags.FLAGS.worker_hosts = hosts
    absl_flags.FLAGS.task_index = task_index
  
  params = benchmark_cnn_kungfu.make_params_from_flags()

  print('INFO: Start copying data from the blob storage into local SSD')
  start = time.time()
  mox.file.copy_parallel(data_url, params.data_dir)
  print('INFO: Complete copy! The copy task takes: ' + str(time.time() - start) + ' seconds')

  params = benchmark_cnn_kungfu.setup(params)
  bench = benchmark_cnn_kungfu.BenchmarkCNN(params)

  tfversion = cnn_util.tensorflow_version_tuple()
  log_fn('TensorFlow:  %i.%i' % (tfversion[0], tfversion[1]))

  bench.print_info()
  bench.run()

  print('INFO: Copy checkpoints to ' + train_url)
  mox.file.copy_parallel(absl_flags.FLAGS.checkpoint_directory, train_url)

  print('INFO: List checkpoints directory ')
  for file_name in os.listdir(absl_flags.FLAGS.checkpoint_directory):
    print(file_name)


if __name__ == '__main__':
  app.run(main)  # Raises error on invalid flags, unlike tf.app.run()
