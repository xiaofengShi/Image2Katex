'''
Filename: LnGru.py
Project: component
File Created: Friday, 14th December 2018 5:17:17 pm
Author: xiaofeng (sxf1052566766@163.com)
--------------------------
Last Modified: Friday, 14th December 2018 5:17:47 pm
Modified By: xiaofeng (sxf1052566766@163.com)
---------------------------
: 2018.06 - 2018 . 
'''

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, LSTMCell, RNNCell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid, tanh


class LNGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, name, num_units, input_size=None, activation=tanh):
        if input_size is not None:
            print("%s: The input_size parameter is deprecated." % self)
        self._name = name
        self._num_units = num_units
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def _LN(self, tensor, scope=None, epsilon=1e-5):
        assert(len(tensor.get_shape()) == 2)
        m, v = tf.nn.moments(tensor, [1], keep_dims=True)
        if not isinstance(scope, str):
            scope = ''
        with tf.variable_scope(scope + 'layer_norm'):
            scale = tf.get_variable('scale',
                                    shape=[tensor.get_shape()[1]],
                                    initializer=tf.constant_initializer(value=1.0))
            shift = tf.get_variable('shift',
                                    shape=[tensor.get_shape()[1]],
                                    initializer=tf.constant_initializer(value=0.))
        _LnInitial = (tensor - m) / tf.sqrt(v + epsilon)

        return _LnInitial * scale + shift

    def __call__(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells.
            inputs: the sequence at each step ,shape is [B,D]
            state: shape conme from pre step ,shape is [B,_num_units]
        """
        with tf.variable_scope(self._name):
            with vs.variable_scope("Gates"):  # Reset gate and update gate.,reuse=True
                # We start with bias of 1.0 to not reset and not update.
                value = tf.layers.dense(
                    inputs=tf.concat(values=[inputs, state], axis=1),
                    units=2 * self._num_units, use_bias=True,
                    kernel_initializer=tf.constant_initializer(value=1.0))
                r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
                r = self._LN(r, scope='r/')
                u = self._LN(u, scope='u/')
                r, u = sigmoid(r), sigmoid(u)
            with vs.variable_scope("Candidate"):
                Cand = tf.layers.dense(
                    inputs=tf.concat(values=[inputs, r * state],
                                     axis=1),
                    units=self._num_units, use_bias=True)
                c_pre = self._LN(Cand,  scope='new_h/')
                c = self._activation(c_pre)
            new_h = u * state + ( 1 - u) * c
        return new_h, new_h


class LNLSTMCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, name, num_units, input_size=None, activation=tanh):
        if input_size is not None:
            print("%s: The input_size parameter is deprecated." % self)
        self._name = name
        self._num_units = num_units
        self._activation = activation

    @property
    def state_size(self):
        return 2*self._num_units

    @property
    def output_size(self):
        return self._num_units

    def _LN(self, tensor, scope=None, epsilon=1e-5):
        assert(len(tensor.get_shape()) == 2)
        m, v = tf.nn.moments(tensor, [1], keep_dims=True)
        if not isinstance(scope, str):
            scope = ''
        with tf.variable_scope(scope + 'layer_norm'):
            scale = tf.get_variable('scale',
                                    shape=[tensor.get_shape()[1]],
                                    initializer=tf.constant_initializer(value=1.0))
            shift = tf.get_variable('shift',
                                    shape=[tensor.get_shape()[1]],
                                    initializer=tf.constant_initializer(value=0.))
        _LnInitial = (tensor - m) / tf.sqrt(v + epsilon)

        return _LnInitial * scale + shift

    def __call__(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells.
            inputs: the sequence at each step ,shape is [B,D]
            state: shape conme from pre step ,shape is [B,_num_units]
        """
        c_tm1, h_tm1 = tf.split(axis=1, num_or_size_splits=2, value=state)
        with tf.variable_scope(self._name):
            with vs.variable_scope("Gates"):  # Reset gate and update gate.,reuse=True
                # We start with bias of 1.0 to not reset and not update.
                value = tf.layers.dense(
                    inputs=tf.concat(values=[inputs, h_tm1],
                                     axis=1),
                    units=4 * self._num_units, use_bias=True,
                    kernel_initializer=tf.orthogonal_initializer(),
                    bias_initializer=tf.constant_initializer(1.0))
                i, f, o, g = array_ops.split(value=value, num_or_size_splits=4, axis=1)

                i = self._LN(i, scope='input/')
                f = self._LN(f, scope='forget/')
                o = self._LN(o, scope='output/')
                g = self._LN(g, scope='instep/')

                i, f, o, g = sigmoid(i), sigmoid(f), sigmoid(o), tanh(g)

            with vs.variable_scope("Candidate"):
                c_pre = f*c_tm1+i*g
                new_cell = self._LN(c_pre, scope='new_cell/')

            new_h = o * self._activation(new_cell)
            new_state = tf.concat(axis=1, values=[new_cell, new_h])

        return new_h, new_state


class MLTAttentionCell(RNNCell):
    """ Attention structure for the MIT model 
        input vector and target vector are all sequences
    """

    def __init__(self, name, num_units, encoder_output, decoder_cell=None, input_size=None):
        if input_size is not None:
            print("%s: The input_size parameter is deprecated." % self)
        self._name = name
        self._num_units = num_units
        if decoder_cell is None:
            self._decoder_cell = GRUCell(num_units=num_units)
        else:
            self._decoder_cell = decoder_cell
        self._encoder_output = encoder_output  # B,L,E_D
        self._max_length = self._encoder_output.get_shape().as_list()[1]

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells.
            inputs: the sequence at each step ,shape is [B,D]
            state: shape conme from pre step ,shape is [B,_num_units]

        """
        hidden = state  # B,D
        with tf.variable_scope(self._name):
            _att = tf.layers.dense(
                inputs=tf.concat(values=[inputs, hidden], axis=1),
                units=self._max_length)  # B,L

            attention_weight = tf.nn.softmax(_att)  # B,L
            attention_weight = tf.expand_dims(attention_weight, axis=1)  # B,1,L

            att_applied = tf.matmul(attention_weight, self._encoder_output)[
                :, 0]  # (B,1,L)*(B,L,D)=(B,1,D)->(B,D)

            output = tf.layers.dense(
                inputs=tf.concat(values=[inputs, att_applied],
                                 axis=1),
                units=self._num_units)  # (B,D)

            output = tf.nn.relu(output)

            output, hidden = self._decoder_cell.__call__(output, hidden)

        return output, hidden
