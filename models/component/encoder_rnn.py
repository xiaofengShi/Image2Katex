'''
Filename: encoder_rnn.py
Project: component
File Created: Friday, 14th December 2018 5:36:20 pm
Author: xiaofeng (sxf1052566766@163.com)
--------------------------
Last Modified: Friday, 14th December 2018 5:37:07 pm
Modified By: xiaofeng (sxf1052566766@163.com)
---------------------------
: 2018.06 - 2018 . 
'''

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from models.component.LnRnn import LNGRUCell, LNLSTMCell
from models.component.rnn_utils import BiLSTM

# from tensorflow.contrib.rnn import GRUCell, LSTMCell


def BiRnn(name, inputs, units):
    """
    Compute recurrent memory states using Bidirectional Long Short-Term Memory units

    :parameters:
        n_in : int ; Dimensionality of input
        n_hid : int ; Dimensionality of hidden state / memory state
        h0_1: vector ; Initial hidden state of forward LSTM
        h0_2: vector ; Initial hidden state of backward LSTM
    """
    # batch_size = tf.shape(inputs)[0]

    with tf.variable_scope(name):

        cell_fw = LNGRUCell(name='fw', num_units=units)
        cell_bw = LNGRUCell(name='bw', num_units=units)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw, cell_bw=cell_bw, inputs=inputs, swap_memory=True, dtype=tf.float32)

    return tf.concat(axis=2, values=[outputs[0], outputs[1]])


def RnnEncoder(name, _image_cnn, _rnn_encoder_dim):
    """  
    Args:
        _image_cnn:the feature come fron cnn, shape is (B,H,W,C)
        _batchsize: must be a specific int number
        _rnn_encoder_dim: num unit of the rnn layer
            encode the feature in the height dimension, H * (B,W,C), 
            it means rnn H times, and each time the input tensor is (B,W,C)
    return:
       encoder out sequence: shape is (B,H*W,2*rnn_encoder_dim)
    """
    batch_size = tf.shape(_image_cnn)[0]
    height = tf.shape(_image_cnn)[1]
    with tf.variable_scope(name):
        def fn(x, i):
            return BiRnn(
                name='encoder_bilstm', inputs=_image_cnn[:, i], units=_rnn_encoder_dim)

        _RnnEncoderOut = tf.scan(fn, tf.range(height), initializer=tf.placeholder(
            shape=(None, None, 2*_rnn_encoder_dim), dtype=tf.float32))

        RnnEncoderOut = tf.reshape(
            tf.transpose(_RnnEncoderOut, [1, 0, 2, 3]),
            [batch_size, -1, 2 * _rnn_encoder_dim])

    return RnnEncoderOut


def RnnEncoderOri(name, _image_cnn, _rnn_encoder_dim):
    """  
    Args:
        _image_cnn:the feature come fron cnn, shape is (B,H,W,C)
        _batchsize: must be a specific int number
        _rnn_encoder_dim: num unit of the rnn layer
            encode the feature in the height dimension, H * (B,W,C), 
            it means rnn H times, and each time the input tensor is (B,W,C)
    return:
       encoder out sequence: shape is (B,H*W,2*rnn_encoder_dim)
    """
    batch_size = tf.shape(_image_cnn)[0]
    height = tf.shape(_image_cnn)[1]
    h0_fw = tf.tile(
        tf.Variable(
            name='enc_fw', initial_value=np.zeros(
                shape=(1, 1, 2 * _rnn_encoder_dim),
                dtype='float32')),
        [batch_size, height, 1])
    h0_bw = tf.tile(
        tf.Variable(
            name='enc_bw', initial_value=np.zeros(
                shape=(1, 1, 2 * _rnn_encoder_dim),
                dtype='float32')),
        [batch_size, height, 1])

    def fn(x, i):
        return BiLSTM(
            name='encoder_bilstm', inputs=_image_cnn[:, i],
            n_hid=_rnn_encoder_dim, h0_fw=h0_fw[:, i],
            h0_bw=h0_bw[:, i])

    _RnnEncoderOut = tf.scan(fn, tf.range(height), initializer=tf.placeholder(
        shape=(None, None, 2*_rnn_encoder_dim), dtype=tf.float32))
    # _Func = tf.make_template('fun', fn)
    # shape is (batch size, rows, columns, features)
    # swap axes so rows are first. map splits tensor on first axis, so fn will be applied to tensors
    # of shape (batch_size,time_steps,feat_size)

    # _RowFist = tf.transpose(_image_cnn, [1, 0, 2, 3])
    # _RnnEncoderOut = tf.map_fn(_Func, _RowFist, dtype=tf.float32)
    # SHAPE IS [batch,h*w,2*rnn_encoder_dim]
    RnnEncoderOut = tf.reshape(
        tf.transpose(_RnnEncoderOut, [1, 0, 2, 3]),
        [batch_size, -1, 2 * _rnn_encoder_dim])

    return RnnEncoderOut
