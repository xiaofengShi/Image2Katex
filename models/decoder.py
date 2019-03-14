'''
Filename: decoder.py
Project: models
File Created: Wednesday, 11th July 2018 3:37:09 pm
Author: xiaofeng (sxf1052566766@163.com)
--------------------------
Last Modified: Sunday, 2nd December 2018 4:09:59 pm
Modified By: xiaofeng (sxf1052566766@163.com)
---------------------------
Copyright: 2018.06 - 2018 OnionMath. OnionMath
'''

from __future__ import division

import tensorflow as tf

from models.component.attention_cell_compile import AttCell
from models.component.decoder_beamsearch import BeamSearchDecoderCell
# from .component.attention_cell_step import AttCell
from models.component.decoder_dynamic import dynamic_decode
from models.component.decoder_greedy import GreedyDecoderCell
from tensorflow.contrib.rnn import GRUCell, LSTMCell
from models.component.LnRnn import LNGRUCell, LNLSTMCell
from models.component.word_embeding import Embedding, embedding_initializer


class DecoderAtt(object):
    def __init__(self, config, vocab):
        self._config = config
        self._vocab = vocab
        self._name = self._config.model.get('decoder_name')
        self._vocabsize = self._vocab.vocab_size
        self._id_end = self._config.dataset.get('id_end')
        self._embeding_dim = self._config.model.get('embeding_dims')
        self._encoder_dim = self._config.model.get('rnn_encoder_dim')
        self._decoder_dim = self._config.model.get('rnn_decoder_dim')
        self._att_dim = self._config.model.get('att_dim')
        assert self._encoder_dim * 2 == self._decoder_dim, \
            "Encoder bilstm out dim is the double encoder dim and it must be equal with decoder dim"

        self._tiles = 1 if self._config.model.decoding == 'greedy' else self._config.model.beam_size

        self._vocab_embeding = tf.get_variable(
            "vocab_embeding", dtype=tf.float32, shape=[self._vocabsize, self._embeding_dim],
            initializer=embedding_initializer())
        self._start_token = tf.squeeze(input=self._vocab_embeding[0, :], name='start_flage')

    def __call__(self, encoder_out, droupout, input_sequence=None):

        self._batch_size = tf.shape(encoder_out)[0]

        with tf.variable_scope(self._name, reuse=False):
            sequence_embeding = Embedding('embeding', self._vocab_embeding, input_sequence)
            # attention cell come from Rnn
            """ Uniform gru cell """
            RnnCell = GRUCell(name='DecoderGru', num_units=self._decoder_dim)
            """ LN gru cell """
            # RnnCell = LNGRUCell(name='DecoderGru', num_units=self._decoder_dim)
            att_cell = AttCell(
                name='AttCell', att_input=encoder_out, cell=RnnCell, n_hid=self._decoder_dim,
                dim_att=self._att_dim, dim_o=self._decoder_dim, dropuout=droupout,
                vacab_size=self._vocabsize)
            # [batch,sequence_length]
            # sequence_length is equal with the input label length
            sequence_length = tf.tile(tf.expand_dims(
                tf.shape(sequence_embeding)[1], 0), [self._batch_size])

            pred_train, _ = tf.nn.dynamic_rnn(
                att_cell, sequence_embeding, initial_state=att_cell.initial_state(),
                sequence_length=sequence_length, dtype=tf.float32, swap_memory=True)
        # evaluating , predict
        with tf.variable_scope(self._name, reuse=True):
            """ uniform gru cell """
            RnnCell = GRUCell(name='DecoderGru', num_units=self._decoder_dim)
            """ LN gru cell """
            # RnnCell = LNGRUCell(name='DecoderGru', num_units=self._decoder_dim)
            att_cell = AttCell(
                name='AttCell', att_input=encoder_out, cell=RnnCell, n_hid=self._decoder_dim,
                dim_att=self._att_dim, dim_o=self._decoder_dim, dropuout=droupout,
                vacab_size=self._vocabsize, tiles=self._tiles)
            if self._config.model.decoding == 'beams_search':
                decoder_cell = BeamSearchDecoderCell(
                    self._vocab_embeding, att_cell, self._batch_size, self._start_token,
                    self._id_end, self._config.model.beam_size,
                    self._config.model.div_gamma, self._config.model.div_prob)
            else:
                decoder_cell = GreedyDecoderCell(
                    self._vocab_embeding, att_cell, self._batch_size, self._start_token,
                    self._id_end)
            pred_validate, _ = dynamic_decode(
                decoder_cell, self._config.model.MaxPredictLength + 1)

        return pred_train, pred_validate
