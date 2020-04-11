

from __future__ import absolute_import, division, print_function

import math

import tensorflow as tf
from six.moves import xrange
from tensorflow.contrib import rnn
import numpy as np


class LSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, name, n_hid, forget_bias=1.0):
        self._n_hid = n_hid
        self._name = name
        self._forget_bias = forget_bias

    @property
    def state_size(self):
        return self._n_hid

    @property
    def output_size(self):
        return self._n_hid

    def __call__(self, inputs, state, scope=None):
        c_tm1, h_tm1 = tf.split(axis=1, num_or_size_splits=2, value=state)

        gates = tf.layers.dense(inputs=tf.concat(axis=1, values=[inputs, h_tm1]),
                                units=4*self._n_hid, name='gates', activation=None)

        i_t, f_t, o_t, g_t = tf.split(axis=1, num_or_size_splits=4, value=gates)

        c_t = tf.nn.sigmoid(f_t + self._forget_bias) * c_tm1 + tf.nn.sigmoid(i_t) * tf.tanh(g_t)
        h_t = tf.nn.sigmoid(o_t) * tf.tanh(c_t)

        new_state = tf.concat(axis=1, values=[c_t, h_t])

        return h_t, new_state


def BiLSTM(name, inputs, n_hid, h0_fw, h0_bw):
    """
    Compute recurrent memory states using Bidirectional Long Short-Term Memory units

    :parameters:
        n_in : int ; Dimensionality of input
        n_hid : int ; Dimensionality of hidden state / memory state
        h0_1: vector ; Initial hidden state of forward LSTM
        h0_2: vector ; Initial hidden state of backward LSTM
    """
    batch_size = tf.shape(inputs)[0]

    with tf.variable_scope(name):

        cell_fw = LSTMCell(name='fw', n_hid=n_hid)
        cell_bw = LSTMCell(name='bw', n_hid=n_hid)

        seq_len = tf.tile(tf.expand_dims(tf.shape(inputs)[1], 0), [batch_size])
        outputs = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                  cell_bw,
                                                  inputs,
                                                  sequence_length=seq_len,
                                                  initial_state_fw=h0_fw,
                                                  initial_state_bw=h0_bw,
                                                  swap_memory=True)

    return tf.concat(axis=2, values=[outputs[0][0], outputs[0][1]])
