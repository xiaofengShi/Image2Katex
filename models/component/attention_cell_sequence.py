'''
File: attention_cell_sequence.py
Project: component
File Created: Friday, 28th December 2018 6:05:05 pm
Author: xiaofeng (sxf1052566766@163.com)
-----
Last Modified: Friday, 28th December 2018 6:50:40 pm
Modified By: xiaofeng (sxf1052566766@163.com>)
-----
 2018.06 - 2018 Latex Math, Latex Math
'''

import collections

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell


# AttentionState = {"att_weight": [], "decoder_out": [], "logits": [], "decoder_state": []}
AttentionState = collections.namedtuple("AttentionState", ("cell_state", "output"))

Attention_weight = list()


class AttCell(RNNCell):
    """ Bahdanau Attention compile for the errorchecker model"""

    def __init__(self, name, attention_in, decoder_cell, n_hid, dim_att, dim_o, dropuout,
                 vacab_size, tiles=1, dtype=tf.float32):
        self._scope_name = name
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        self._encoder_sequence = attention_in
        
        if isinstance(attention_in, tuple):
            self._encoder_sequence = tf.concat(attention_in, 2)

        self._cell = decoder_cell   # decoder rnn cell
        self._n_hid = n_hid  # decoder num_unit D_DIM
        self._dim_att = dim_att   # Attention size，计算的中间变量，一般可以选择输入的_encoder_sequence相同的维度
        self._dim_o = dim_o   # the dim of output, same with the param: n_hid
        self._dropout = dropuout  # droupout rate
        self._vacab_size = vacab_size  # the vocabulary size of the decoder, same with the machine translation model

        # in the decoder stage, if use the beamsearch trick, the tiles is needed, default value is 1 for the greedy trick
        self._tiles = tiles
        self._dtype = dtype  # default is tf.float32
        self._length = tf.shape(self._encoder_sequence)[1]  # length of the input sequence
        self._en_dim = self._encoder_sequence.shape[2].value  # dims of the encoder

        self._state_size = AttentionState(self._n_hid, self._dim_o)

        self._att_seq = tf.layers.dense(
            inputs=self._encoder_sequence, units=self._dim_att, use_bias=False, name="att_img")  # B,L,dim_att

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        # beacause in the function the return is logits,so the size is vocab_size
        return self._vacab_size

    @property
    def output_dtype(self):
        return self._dtype

    def _CalStateBasedSeq(self, name, dim):
        """Returns initial state of dimension specified by dim"""
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            # (B*T,L,E_DIM) -->(B*T,E_DIM)
            img_mean = tf.reduce_mean(self._encoder_sequence, axis=1)
            W = tf.get_variable("W_{}_0".format(name), shape=[self._en_dim, dim])
            b = tf.get_variable("b_{}_0".format(name), shape=[1, dim])
            h = tf.tanh(tf.matmul(img_mean, W) + b)

            return h

    def initial_state(self):
        """ setting initial state  and output """
        initial_states = self._CalStateBasedSeq('init_state', self._n_hid)  # (B,HID)
        initial_out = self._CalStateBasedSeq('init_out', self._dim_o)  # (B,DIM_O)
        return AttentionState(initial_states, initial_out)

    def _cal_att(self, hid_cur):
        with tf.variable_scope('att_cal'):
            if self._tiles > 1:
                _encoder_sequence = tf.expand_dims(self._encoder_sequence, axis=1)  # (B,1,L,E_DIM)
                _encoder_sequence = tf.tile(_encoder_sequence, multiples=[
                    1, self._tiles, 1, 1])  # (B,T,L,E_DIM)
                _encoder_sequence = tf.reshape(
                    _encoder_sequence, shape=[-1, self._length, self._en_dim])  # (B*T,L,E_DIM)

                _att_seq = tf.expand_dims(self._att_seq, axis=1)  # B,1,L,dim_att
                _att_seq = tf.tile(_att_seq, multiples=[1, self._tiles, 1, 1])
                _att_seq = tf.reshape(
                    _att_seq, shape=[-1, self._length, self._dim_att])  # (B*T,L,dim_att)
            else:
                _att_seq = self._att_seq
                _encoder_sequence = self._encoder_sequence
            # computes attention over the hidden vector
            # hid_cur shape is  [ B,num_units]
            # att_h [B,dim_att]
            att_h = tf.layers.dense(inputs=hid_cur, units=self._dim_att, use_bias=False)
            # sums the two contributions
            # att_h --> [B,1,dim_att]
            att_h = tf.expand_dims(att_h, axis=1)
            # Computes the score for the Bahdanau style
            # _att_seq contains the full encoder output, shape is [batch，L, _dim_att]
            # att_h contains the current hiddent of the deocder, shape is [B,1,dim_att]
            att = tf.tanh(_att_seq + att_h)  # shape [B,L,dim_att]
            # computes scalar product with beta vector
            # works faster with a matmul than with a * and a tf.reduce_sum
            # For each of the timestamps its vector of size A from `att` is reduced with `att_beta` vector
            att_beta = tf.get_variable("att_beta", shape=[self._dim_att, 1], dtype=tf.float32)
            # att_flat shape is [B*L,dim_att]
            att_flat = tf.reshape(att, shape=[-1, self._dim_att])
            # computes score
            e = tf.matmul(att_flat, att_beta)  # shape is [B*L,1]
            e = tf.reshape(e, shape=[-1, self._length])  # shape is [B,L]
            # computes attention weights
            attention = tf.nn.softmax(e)  # shape is (B,L)
            _att = tf.expand_dims(attention, axis=-1)  # (B,L,1)
            # computes the contex vector with the attention and encoder_sequence
            contex = tf.reduce_sum(_att * _encoder_sequence, axis=1)  # [B,L,1]*[B,L,E]=(B,E)

            return attention, contex

    def step(self, embeding, attention_cell_state):
        """
        Args:
            embeding: shape is (B,EMBEDING_DIM)
            attention_cell_state: state from previous step comes from AttentionState 
        """
        _initial_state, output_tm1 = attention_cell_state
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope, initializer=tf.orthogonal_initializer()):
            x = tf.concat([embeding, output_tm1], axis=-1)
            # compute current hidden and cell states
            new_hid, new_cell_state = self._cell.__call__(inputs=x, state=_initial_state)
            _attention, contex = self._cal_att(new_hid)

            def _debug_att(val):
                global Attention_weight
                Attention_weight = []
                Attention_weight += [val]
                return False

            print_func = tf.py_func(_debug_att, [_attention], [tf.bool])
            with tf.control_dependencies(print_func):
                _attention = tf.identity(_attention, name='Attention_weight')
            o_W_c = tf.get_variable("o_W_c", dtype=tf.float32,
                                    shape=(self._en_dim, self._n_hid))
            o_W_h = tf.get_variable("o_W_h", dtype=tf.float32,
                                    shape=(self._n_hid, self._dim_o))
            new_o = tf.tanh(tf.matmul(new_hid, o_W_h) + tf.matmul(contex, o_W_c))
            new_o = tf.nn.dropout(new_o, self._dropout)
            y_W_o = tf.get_variable("y_W_o", dtype=tf.float32,
                                    shape=(self._dim_o, self._vacab_size))
            # logits for current step
            # shape is [B,vocabsize] for each size
            logits = tf.matmul(new_o, y_W_o)
            new_state = AttentionState(new_cell_state, new_o)
            return logits, new_state

    def __call__(self, _inputs, _state):
        """
        The dynamic rnn function will use this call function to calculate step by step
        Args:
            inputs: the embedding of the previous word for training only，decoder sequence 
            state: (AttentionState) (h,c, o) where h is the hidden state and
                o is the vector used to make the prediction of
                the previous word
        """
        logits, state = self.step(_inputs, _state)
        return (logits, state)
