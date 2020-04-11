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

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, RNNCell
import collections
import numpy as np

'''
Attentional Decoder as proposed in HarvardNLp paper (https://arxiv.org/pdf/1609.04938v1.pdf)
'''

# AttentionState = {"att_weight": [], "decoder_out": [], "logits": [], "decoder_state": []}
AttentionState = collections.namedtuple("AttentionState", ("cell_state", "output"))

Attention_weight = []


class AttCell(RNNCell):
    def __init__(
            self, att_input, n_hid, vacab_size, batch_size, tiles, forget_bias=1.0,
            dtype=tf.float32):
        self._encoder_output = att_input  # img-cnn=rnn之后得到 [B,HW,E_DIM]
        self._n_hid = n_hid  # decoder的隐藏节点数 D_DIM
        self._forget_bias = forget_bias  # 遗忘门的偏执
        self._vacab_size = vacab_size  # 字典大小
        self._dtype = dtype
        self._batch_size = batch_size
        self._tiles = tiles
        self._n_regions = tf.shape(self._encoder_output)[1]  # HW
        self._n_channels = self._encoder_output.shape[2].value  # E_DIM
        if self._tiles > 1:
            self._encoder_output = tf.expand_dims(self._encoder_output, axis=1)  # (B,1,HW,E_DIM)
            self._encoder_output = tf.tile(self._encoder_output, multiples=[
                                           1, self._tiles, 1, 1])  # (B,T,HW,E_DIM)
            self._encoder_output = tf.reshape(
                self._encoder_output, shape=[-1, self._n_regions, self._n_channels])  # (B*D,HW,E_DIM)

    @property
    def state_size(self):
        return self._n_hid

    @property
    def output_size(self):
        return self._vacab_size  # beacause in the function the return is logits,so the size is vocab_size

    @property
    def output_dtype(self):
        return self._dtype

    def initial_state(self):
        """ setting initial state  and output """
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            # B,2*D_DIM
            initial_cell_state = tf.tile(
                tf.Variable(
                    name='initiate_cell_state', initial_value=np.zeros(
                        (1, 2 * self._n_hid),
                        np.float32)),
                [self._batch_size, 1])
            # B,D_DIM
            initial_out = tf.tile(tf.Variable(
                name='initiate_output', initial_value=np.zeros(
                    (1, self._n_hid),
                    np.float32)), [self._batch_size, 1])

        return AttentionState(initial_cell_state, initial_out)

    def step(self, embeding, attention_cell_state):
        """
        Args:
            embeding: shape is (B,EM_DIM)
            attention_cell_state: state from previous step comes from AttentionState 
        """
        initial_state, output_tm1 = attention_cell_state
        # B,D_DIM
        h_tm1, c_tm1 = tf.split(axis=1, num_or_size_splits=2, value=initial_state)

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            gates = tf.layers.dense(
                inputs=tf.concat(axis=1, values=[embeding, output_tm1]),
                units=4*self._n_hid, name='gates', activation=None)

            i_t, f_t, o_t, g_t = tf.split(axis=1, num_or_size_splits=4, value=gates)  # B*HID

            c_t = tf.nn.sigmoid(f_t) * c_tm1 + tf.nn.sigmoid(i_t) * tf.tanh(g_t)  # memory cell
            h_t = tf.nn.sigmoid(o_t) * tf.tanh(c_t)  # hidden_output (batch,_n_hid)=(B,D)

            # (B,D,1)
            target_tmp = tf.expand_dims(tf.layers.dense(
                name='target_tmp', inputs=h_t, units=self._n_hid, use_bias=False), axis=2)

            #  (B, H*W, D) * (B, D, 1)=(B,H*W,1)
            # ATTENTION_WEIGHT shape (B,H*W)
            a_t = tf.nn.softmax(tf.matmul(self._encoder_output, target_tmp)
                                [:, :, 0], name='Attention_weight')

            def _debug_bkpt(val):
                global Attention_weight
                Attention_weight += [val]
                return False

            debug_print_op = tf.py_func(_debug_bkpt, [a_t], [tf.bool])
            with tf.control_dependencies(debug_print_op):
                a_t = tf.identity(a_t, name='a_t_debug')
            # (B, 1, H*W)
            a_t = tf.expand_dims(a_t, 1)
            # (B,1,H*W)*(B,H*W,D)=(B,1,D)-->(B,D)
            z_t = tf.matmul(a_t, self._encoder_output)[:, 0]  # (B,D)
            # (B,D)
            new_output = tf.layers.dense(
                name='output', inputs=tf.concat(axis=1, values=[h_t, z_t]),
                units=self._n_hid, use_bias=False, activation=tf.tanh)
            # (B,2*HID)
            _new_state = tf.concat(axis=1, values=[h_t, c_t])

            new_state = AttentionState(_new_state, new_output)

            y_W_o = tf.get_variable("y_W_o", dtype=tf.float32,
                                    shape=(self._n_hid, self._vacab_size))

            logits = tf.matmul(new_output, y_W_o)

            # logits = tf.layers.dense(name='logits', inputs=new_output,
            #                          units=self._vacab_size, activation=None, use_bias=True)

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
