'''
Filename: encoder_cnn.py
Project: component
File Created: Friday, 14th December 2018 3:32:36 pm
Author: xiaofeng (sxf1052566766@163.com)
--------------------------
Last Modified: Friday, 14th December 2018 3:33:12 pm
Modified By: xiaofeng (sxf1052566766@163.com)
---------------------------
: 2018.06 - 2018 . 
'''


from __future__ import absolute_import, division, print_function

import math

import tensorflow as tf
from six.moves import xrange


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


def _var_random(name, shape, regularizable=False):

    v = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    if regularizable:
        with tf.name_scope(name + '/Regularizer/'):
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(v))
    return v


def ConvElus(incoming, num_filters, filter_size, name, strides=1,
             padding_type='SAME'):

    # conved = tf.layers.conv2d(
    #     inputs=incoming, filters=num_filters, kernel_size=filter_size, strides=strides,
    #     padding=padding_type, name=name, kernel_initializer=tf.contrib.layers.xavier_initializer(),
    #     activation=tf.nn.elu)
    num_filters_from = incoming.get_shape().as_list()[3]
    _strides = (1, strides, strides, 1)
    with tf.variable_scope(name):
        conv_W = _var_random(
            'W', tuple(filter_size) + (num_filters_from, num_filters),
            regularizable=True)

        conved = tf.nn.conv2d(incoming, conv_W, strides=_strides,
                              padding=padding_type, data_format='NHWC')

        return tf.nn.elu(conved)


def MaxPool(incoming, name, kernel_size=(2, 2), stride_size=(2, 2), padding_type='SAME'):

    # pooled = tf.layers.max_pooling2d(
    #     inputs=incoming, pool_size=kernel_size, strides=stride_size, padding=padding_type,
    #     name=name)
    # return pooled
    
    _kerne = (1, kernel_size[0], kernel_size[1], 1)
    _stride = (1, stride_size[0], stride_size[1], 1)
    with tf.variable_scope(name):
        return tf.nn.max_pool(incoming, ksize=_kerne, strides=_stride, padding=padding_type)


def ImageCNN(name, img, config):

    print('input_tensor dim: {}'.format(img.get_shape()))
    img = tf.cast(img, tf.float32)
    net = tf.add(img, (-128.0))
    net = tf.multiply(net, (1/128.0))
    end_layers = {}
    with tf.variable_scope(name):
        # conv + max pool -> /2
        net = ConvElus(img, 64, (3, 3), 'conv1_conv1')
        net = MaxPool(net, 'conv1_pool1', (2, 2), (2, 2))

        net = ConvElus(net, 128, (3, 3), 'conv2_conv2')
        net = MaxPool(net, 'conv2_pool2', (2, 2), (2, 2))

        net = ConvElus(net, 256, (3, 3), 'conv3_conv3')
        net = ConvElus(net, 256, (3, 3), 'conv3_conv4')

        if config.model.encoder_cnn == "vanilla":
            # net = tf.layers.max_pooling2d(net, (2, 1), (2, 1), "SAME")
            net = MaxPool(net, 'conv4_pool1',  (2, 1), (2, 1))

        net = ConvElus(net, 512, (3, 3), 'conv4_conv1')
        end_layers['conv4_conv1'] = net

        if config.model.encoder_cnn == "vanilla":
            net = MaxPool(net, 'conv5_pool1', (1, 2), (1, 2))

        if config.model.encoder_cnn == "cnn":
            # conv with stride /2 (replaces the 2 max pool)
            net = ConvElus(net, 512, (2, 4), 'conv6_conv1', strides=2)
        # conv
        # the padding is VALID
        net = ConvElus(net, 512, (3, 3), 'conv7_conv1', padding_type='VALID')
        end_layers['conv7_conv1'] = net

        if config.model.positional_embeddings:
            # from tensor2tensor lib - positional embeddings
            # add location
            net = add_timing_signal_nd(net)
            end_layers['positional_embeddings_layers'] = net

    print('encoder_output_imgdim: {}'.format(net.get_shape()))

    return net, end_layers


