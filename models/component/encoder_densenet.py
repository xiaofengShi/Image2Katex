'''
Filename: encoder_densenet.py
Project: models
File Created: Sunday, 9th December 2018 8:37:38 pm
Author: xiaofeng (sxf1052566766@163.com)
--------------------------
Last Modified: Sunday, 9th December 2018 8:38:04 pm
Modified By: xiaofeng (sxf1052566766@163.com)
---------------------------
Copyright: 2018.06 - 2018 OnionMath. OnionMath
'''
from __future__ import absolute_import, division, print_function

import math

import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm, xavier_initializer

# tf.nn.batch_normalization()

def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(
            inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride,
            padding='SAME', kernel_initializer=xavier_initializer(),
            kernel_regularizer=tf.nn.l2_normalize)
        return network


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


def Drop_out(x, rate, training):
    return tf.layers.dropout(inputs=x, rate=rate, training=training)


def Relu(x):
    return tf.nn.relu(x)


def Average_pooling(x, pool_size=[2, 2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(
        inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Concatenation(layers):
    return tf.concat(values=layers, axis=3)


def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e4):
    # NOTE sorce https://github.com/tensorflow/tensor2tensor/blob/37465a1759e278e8f073cd04cd9b4fe377d3c740/tensor2tensor/layers/common_attention.py
    """Adds a bunch of sinusoids of different frequencies to a Tensor.

    Each channel of the input Tensor is incremented by a sinusoid of a difft
    frequency and phase in one of the positional dimensions.

    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.

    The use of relative position is possible because sin(a+b) and cos(a+b) can
    be experessed in terms of b, sin(a) and cos(a).

    x is a Tensor with n "positional" dimensions, e.g. one dimension for a
    sequence or two dimensions for an image

    We use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels // (n * 2). For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.

    Args:
        x: a Tensor with shape [batch, d1 ... dn, channels]
        min_timescale: a float
        max_timescale: a float

    Returns:
        a Tensor the same shape as x.

    """
    static_shape = x.get_shape().as_list()  # [B,H,W,C]
    num_dims = len(static_shape) - 2  # 2
    channels = tf.shape(x)[-1]  # C
    num_timescales = channels // (num_dims * 2)  # C//2*2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    for dim in xrange(num_dims):
        length = tf.shape(x)[dim + 1]
        position = tf.to_float(tf.range(length))
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
            inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 1) * 2 * num_timescales
        signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
        for _ in xrange(1 + dim):
            signal = tf.expand_dims(signal, 0)
        for _ in xrange(num_dims - 1 - dim):
            signal = tf.expand_dims(signal, -2)
        x += signal
    return x


class DenseNet(object):
    def __init__(self, x, filters_growth, dropout_rate, filters_out_nums=32, training=True,
                 nb_layers=None):
        """  
        Args:
            x: image,shape is [b,h,w,c]
            filters_growth: filter nums of each step
            dropout_rate: droupout rate for training
            training: bool true or false
        """
        self.filters = filters_growth
        self.training = training
        self.dropout_rate = dropout_rate
        if nb_layers is None:
            self.nb_layers = [6, 12, 48]
        else:
            self.nb_layers = nb_layers
        self.filters_out_nums = filters_out_nums
        self.cnn_out = self.Dense_net(x)

    def bottleneck_layer(self, x, scope):
        """  
        bottleneck layer
        this layer will generate the small feature layer,
         out layer filter nums is is setting filters
        """
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1, 1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3, 3], layer_name=scope+'_conv2')
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)
            return x

    def transition_layer(self, input_x, scope):
        """  
        In the transition layer, downsample the feature layer  and rate is 2
        structure is :
            BN+RELU+CONV1+DROPUOUT+AVGPool2
        """
        with tf.name_scope(scope):
            x = Batch_Normalization(input_x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1, 1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2, 2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        """ denseblock which nb_layers bottle_layers """
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = Concatenation(layers_concat)

            return x

    def Dense_net(self, input_x):

        x = conv_layer(input_x, filter=2 * self.filters,
                       kernel=[5, 5], stride=2, layer_name='conv0')

        for idx in range(len(self.nb_layers)):
            x = self.dense_block(
                input_x=x, nb_layers=self.nb_layers[idx],
                layer_name='dense_%d' % idx)
            x = self.transition_layer(input_x=x, scope='trans_%d' % idx)

        x = self.dense_block(input_x=x, nb_layers=self.filters_out_nums, layer_name='dense_final')

        # 100 Layer
        x = Batch_Normalization(x, training=self.training, scope='dense_batch')
        x = Relu(x)
        """ add timing signal  """
        x = add_timing_signal_nd(x)
        return x
