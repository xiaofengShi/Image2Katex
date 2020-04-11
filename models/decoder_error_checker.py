'''
File: decoder_error_checker.py
Project: models
File Created: Saturday, 29th December 2018 3:07:21 pm
Author: xiaofeng (sxf1052566766@163.com)
-----
Last Modified: Saturday, 29th December 2018 3:07:45 pm
Modified By: xiaofeng (sxf1052566766@163.com>)
-----
 2018.06 - 2018 Latex Math, Latex Math
'''


from __future__ import division

import tensorflow as tf

from models.component.attention_cell_sequence import AttCell
from models.component.decoder_beamsearch import BeamSearchDecoderCell
from models.component.decoder_greedy import GreedyDecoderCell
from models.component.decoder_dynamic import dynamic_decode
from tensorflow.contrib.rnn import GRUCell
from models.component.LnRnn import LNGRUCell, LNLSTMCell
from models.component.word_embeding import Embedding, embedding_initializer


class DecoderAtt(object):
    """ Decoder section of the errorchecker model """

    def __init__(self, config, vocab):
        self._config = config
        self._vocab = vocab
        self._name = self._config.model.get('errche_decoder_name')
        # vocabulary size of the target sequence
        self._targ_voc = self._vocab.errche_vocab_size_targ
        # index of the END token  in the vocabulary
        self._id_end = self._config.dataset.get('id_end')
        # embeding dim of the target sequence
        self._embeding_dim_targ = self._config.model.get('errche_embeding_dims_target')
        # dim of encoder
        self._rnn_encoder_dim = self._config.model.get('errche_rnn_encoder_dim')
        # dim of decoder
        self._rnn_decoder_dim = self._config.model.get('errche_rnn_decoder_dim')
        # dim of attention
        self._att_dim = self._config.model.get('att_dim')
        assert self._rnn_encoder_dim * 2 == self._rnn_decoder_dim, \
            "Encoder BiRnn out dim is the double encoder dim and it must be equal with decoder dim"

        self._tiles = 1 if self._config.model.decoding == 'greedy' else self._config.model.beam_size

        self._embedding_table_traget = tf.get_variable(
            "targ_vocab_embeding", dtype=tf.float32, shape=[self._targ_voc, self._embeding_dim_targ],
            initializer=embedding_initializer())
        self._start_token = tf.squeeze(
            input=self._embedding_table_traget[0, :],
            name='targ_start_flage')

    def __call__(self, encoder_out, droupout, input_sequence=None):

        self._batch_size = tf.shape(encoder_out)[0]

        with tf.variable_scope(self._name, reuse=False,initializer=tf.orthogonal_initializer()):
            sequence_embeding = Embedding('embeding', self._embedding_table_traget, input_sequence)
            # attention cell come from Rnn
            """ Uniform gru cell """
            # RnnCell = GRUCell(name='DecoderGru', num_units=self._rnn_decoder_dim)
            """ LN gru cell """
            RnnCell = LNGRUCell(name='DecoderGru', num_units=self._rnn_decoder_dim)
            att_cell = AttCell(
                name='AttCell', attention_in=encoder_out, decoder_cell=RnnCell,
                n_hid=self._rnn_decoder_dim, dim_att=self._att_dim, dim_o=self._rnn_decoder_dim,
                dropuout=droupout, vacab_size=self._targ_voc)
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
            # RnnCell = GRUCell(name='DecoderGru', num_units=self._rnn_decoder_dim)
            """ LN gru cell """
            RnnCell = LNGRUCell(name='DecoderGru', num_units=self._rnn_decoder_dim)
            att_cell = AttCell(
                name='AttCell', attention_in=encoder_out, decoder_cell=RnnCell,
                n_hid=self._rnn_decoder_dim, dim_att=self._att_dim, dim_o=self._rnn_decoder_dim,
                dropuout=droupout, vacab_size=self._targ_voc, tiles=self._tiles)
            if self._config.model.decoding == 'beams_search':
                decoder_cell = BeamSearchDecoderCell(
                    self._embedding_table_traget, att_cell, self._batch_size, self._start_token,
                    self._id_end, self._config.model.beam_size,
                    self._config.model.div_gamma, self._config.model.div_prob)
            else:
                decoder_cell = GreedyDecoderCell(
                    self._embedding_table_traget, att_cell, self._batch_size, self._start_token,
                    self._id_end)
            pred_validate, _ = dynamic_decode(
                decoder_cell, self._config.model.MaxPredictLength + 1)

        return pred_train, pred_validate
