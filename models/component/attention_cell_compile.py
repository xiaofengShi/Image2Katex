'''
Filename: attention_cell.py
Project: tflib
File Created: Sunday, 2nd December 2018 2:25:07 pm
Author: xiaofeng (sxf1052566766@163.com)
--------------------------
Last Modified: Sunday, 2nd December 2018 2:26:37 pm
Modified By: xiaofeng (sxf1052566766@163.com)
---------------------------
: 2018.06 - 2018 . 
'''

import collections

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, LSTMCell, RNNCell

'''
Attentional Decoder as proposed in HarvardNLp paper (https://arxiv.org/pdf/1609.04938v1.pdf)
'''

# AttentionState = {"att_weight": [], "decoder_out": [], "logits": [], "decoder_state": []}
AttentionState = collections.namedtuple("AttentionState", ("cell_state", "output"))

Attention_weight = list()


class AttCell(RNNCell):
    def __init__(
            self, name, att_input, cell, n_hid, dim_att, dim_o, dropuout, vacab_size, tiles=1,
            dtype=tf.float32):
        self._scope_name = name
        self._encoder_sequence = att_input  # img-cnn=rnn之后得到 [B,HW,E_DIM]
        self._cell = cell   # decoder rnn cell
        self._n_hid = n_hid  # decoder的隐藏节点数 D_DIM
        self._dim_att = dim_att   # Attention 维度，计算的中间变量，一般可以选择输入的_encoder_sequence相同的维度
        self._dim_o = dim_o   # rnn输出的维度，该维度与rnn设置的num_unit相同
        self._dropout = dropuout  # droupout 比例
        self._vacab_size = vacab_size  # 创建的词典包含单词数量，
        self._dtype = dtype
        self._tiles = tiles    # beam search时需要这个变量,在训练阶段，默认tiles为1，在测试阶段根据测试的策略设置tiles为1或5
        self._n_regions = tf.shape(self._encoder_sequence)[1]  # HW
        self._n_channels = self._encoder_sequence.shape[2].value  # E_DIM

        # self._state_size = AttentionState(self._cell._state_size, self._dim_o)
        self._state_size = AttentionState(self._n_hid, self._dim_o)

        self._att_img = tf.layers.dense(
            inputs=self._encoder_sequence, units=self._dim_att, use_bias=False, name="att_img")  # B,HW,dim_att

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._vacab_size  # beacause in the function the return is logits,so the size is vocab_size

    @property
    def output_dtype(self):
        return self._dtype

    def initial_cell_state(self, cell):
        _states_0 = []
        for hidden_name in cell._state_size._fields:
            hidden_dim = getattr(cell._state_size, hidden_name)
            h = self._CalStateBasedSeq(hidden_name, hidden_dim)
            _states_0.append(h)

        initial_state_cell = type(cell.state_size)(*_states_0)

        return initial_state_cell

    def _CalStateBasedSeq(self, name, dim):
        """Returns initial state of dimension specified by dim"""
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            # (B*T,HW,E_DIM) -->(B*T,1E_DIM)
            img_mean = tf.reduce_mean(self._encoder_sequence, axis=1)
            W = tf.get_variable("W_{}_0".format(name), shape=[self._n_channels, dim])
            b = tf.get_variable("b_{}_0".format(name), shape=[1, dim])
            h = tf.tanh(tf.matmul(img_mean, W) + b)

            return h

    def initial_state(self):
        """ setting initial state  and output """
        initial_states = self._CalStateBasedSeq('init_state', self._n_hid)  # (B,HID)
        # initial_states = self.initial_cell_state(self._cell)  # batch,
        initial_out = self._CalStateBasedSeq('init_out', self._dim_o)  # (B,DIM_O)

        return AttentionState(initial_states, initial_out)

    def _cal_att(self, hid_cur):

        with tf.variable_scope('att_cal'):
            if self._tiles > 1:
                _encoder_sequence = tf.expand_dims(
                    self._encoder_sequence, axis=1)  # (B,1,HW,E_DIM)
                _encoder_sequence = tf.tile(_encoder_sequence, multiples=[
                    1, self._tiles, 1, 1])  # (B,T,HW,E_DIM)
                _encoder_sequence = tf.reshape(
                    _encoder_sequence, shape=[-1, self._n_regions, self._n_channels])  # (B*T,HW,E_DIM)

                _att_img = tf.expand_dims(self._att_img, axis=1)  # batch,1,HW,dim_att
                _att_img = tf.tile(_att_img, multiples=[1, self._tiles, 1, 1])
                _att_img = tf.reshape(_att_img, shape=[-1, self._n_regions,
                                                       self._dim_att])
            else:
                _att_img = self._att_img
                _encoder_sequence=self._encoder_sequence
            # computes attention over the hidden vector
            # h [ batch,num_units]
            # att_h [batch,dim_att]
            att_h = tf.layers.dense(inputs=hid_cur, units=self._dim_att, use_bias=False)
            # sums the two contributions
            # att_h --> [batch,1,dim_att]
            att_h = tf.expand_dims(att_h, axis=1)
            # att_img [batch，h*w, _dim_att]
            # att_h [batch,1,dim_att]
            # att shape is [batch,h*w,dim_att]
            att = tf.tanh(_att_img + att_h)

            # computes scalar product with beta vector
            # works faster with a matmul than with a * and a tf.reduce_sum
            att_beta = tf.get_variable("att_beta", shape=[self._dim_att, 1],
                                       dtype=tf.float32)
            # att_flat shape is [batch*h*w,dim_att]
            att_flat = tf.reshape(att, shape=[-1, self._dim_att])
            # [batch*h*w,1]
            e = tf.matmul(att_flat, att_beta)
            # [batch,h*w]
            e = tf.reshape(e, shape=[-1, self._n_regions])
            # compute weights
            # (B,HW)
            attention = tf.nn.softmax(e)
            # B,HW,1
            _att = tf.expand_dims(attention, axis=-1)
            # [B,HW,1]*[B,HW,C]
            # CONTEX SHAPE IS [B,C]
            contex = tf.reduce_sum(_att * _encoder_sequence, axis=1)

            return attention, contex

    def step(self, embeding, attention_cell_state):
        """
        Args:
            embeding: shape is (B,EM_DIM)
            attention_cell_state: state from previous step comes from AttentionState 
        """
        _initial_state, output_tm1 = attention_cell_state
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope, initializer=tf.orthogonal_initializer()):
            x = tf.concat([embeding, output_tm1], axis=-1)
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
            # # B,HW,1
            # attention = tf.expand_dims(_attention, axis=-1)
            # # [B,HW,1]*[B,HW,C]
            # # CONTEX SHAPE IS [B,C]
            # contex = tf.reduce_sum(attention * self._encoder_sequence, axis=1)

            o_W_c = tf.get_variable("o_W_c", dtype=tf.float32,
                                    shape=(self._n_channels, self._n_hid))

            o_W_h = tf.get_variable("o_W_h", dtype=tf.float32,
                                    shape=(self._n_hid, self._dim_o))

            new_o = tf.tanh(tf.matmul(new_hid, o_W_h) + tf.matmul(contex, o_W_c))

            new_o = tf.nn.dropout(new_o, self._dropout)

            y_W_o = tf.get_variable("y_W_o", dtype=tf.float32,
                                    shape=(self._dim_o, self._vacab_size))
            # logits for current step
            # shape is [batch_size,vocabsize] for each size
            logits = tf.matmul(new_o, y_W_o)

            new_state = AttentionState(new_cell_state, new_o)

            return logits, new_state

    def __call__(self, _inputs, _state):
        """
        The dynamic rnn function will use this call function to calculate step by step
        Args:
            inputs: the embedding of the previous word for training only
            state: (AttentionState) (h,c, o) where h is the hidden state and
                o is the vector used to make the prediction of
                the previous word
        """
        logits, state = self.step(_inputs, _state)
        return (logits, state)
