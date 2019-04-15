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

"""Lenet model configuration.

References:
  LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick Haffner
  Gradient-based learning applied to document recognition
  Proceedings of the IEEE (1998)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models import model


class Lenet5Model(model.CNNModel):
  """Lenet5."""

  def __init__(self, params=None):
    super(Lenet5Model, self).__init__('lenet5', 28, 32, 0.005, params=params)

  def add_inference(self, cnn):
    # Note: This matches TF's MNIST tutorial model
    cnn.conv(32, 5, 5)
    cnn.mpool(2, 2)
    cnn.conv(64, 5, 5)
    cnn.mpool(2, 2)
    cnn.reshape([-1, 64 * 7 * 7])
    cnn.affine(512)


class LenetMNISTModel(model.Model):
  def __init__(self):
    super(LenetMNISTModel, self).__init__('lenetmnist', 28, 32, 0.005)

  def skip_final_affine_layer(self):
    return True

  def add_inference(self, cnn):
    print("LENET> input is %s" % cnn.top_layer)

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    # conv1 = tf.layers.conv2d(
    #  inputs=input_layer,
    #  filters=32,
    #  kernel_size=[5, 5],
    #  padding="same",
    #  activation=tf.nn.relu)
    cnn.cbow_conv(32, [5, 5], padding="same", activation=tf.nn.relu,
                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                  bias_initializer=tf.constant_initializer(value=0))

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    cnn.cbow_mpool([2, 2], 2, padding="same")

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    # conv2 = tf.layers.conv2d(
    #  inputs=pool1,
    #  filters=64,
    #  kernel_size=[5, 5],
    #  padding="same",
    #  activation=tf.nn.relu)
    cnn.cbow_conv(64, [5, 5], padding="same", activation=tf.nn.relu,
                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                  bias_initializer=tf.constant_initializer(value=0.1))

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    cnn.cbow_mpool([2, 2], 2, padding="same")

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    flattened = tf.reshape(cnn.top_layer, [-1, 7 * 7 * 64])
    cnn.top_layer = flattened

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    # dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    cnn.cbow_dense(1024, activation=tf.nn.relu,
                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                   bias_initializer=tf.constant_initializer(value=0.1))

    # Add dropout operation; 0.6 probability that element will be kept
    # dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    cnn.cbow_dropout(rate=0.4)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    # logits = tf.layers.dense(inputs=dropout, units=10)
    cnn.cbow_dense(10,
                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                   bias_initializer=tf.constant_initializer(value=0.1))
