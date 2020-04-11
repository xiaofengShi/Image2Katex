'''
File: encoder_error_checker.py
Project: models
File Created: Saturday, 29th December 2018 2:08:44 pm
Author: xiaofeng (sxf1052566766@163.com)
-----
Last Modified: Saturday, 29th December 2018 3:05:33 pm
Modified By: xiaofeng (sxf1052566766@163.com>)
-----
 2018.06 - 2018 Latex Math, Latex Math
'''


from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell

from models.component.error_checker_modules import (encoder_cbhg, post_cbhg,
                                                    prenet)
from models.component.LnRnn import LNGRUCell
from models.component.word_embeding import Embedding, embedding_initializer


class Encoder(object):

    """ Encoder section of the errorchecker model

    Class with a __call__ method that applies encoding to the source sequence"""

    def __init__(self, config, vocab, trainable):
        self._config = config
        self._vocab = vocab
        self._name = self._config.model.get('errche_encoder_name')
        # dim of the encoder, due to the birnn, the output of the encoder is 2*encoder_dim
        self._rnn_encoder_dim = self._config.model.get('errche_rnn_encoder_dim')
        self._embeding_dim_source = self._config.model.get('errche_embeding_dims_source')
        # vocabulary size of the source sequecn
        self._source_voc = self._vocab.errche_vocab_size_source
        self.is_training = trainable

        self._embedding_table_source = tf.get_variable(
            "source_vocab_embeding", dtype=tf.float32,
            shape=[self._source_voc, self._embeding_dim_source],
            initializer=embedding_initializer())

    def __call__(self, inputs, droupout, prenet_flage=True):
        """Applies encoder for the input sequence

        Args:
            inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
                steps in the input time series, and values are character IDs
            droupout: In the train mode, the droupout less than 1.0, and in other mode, the droupout is 1.0, 
                this is the keep_prob between the layters
            prenet_flage: whether use the prenet to downsample the inputu embedings
        Returns:
            encoder_out: the encoded source sequence, shape =[N,T_in,2*E_Dim]
        """
        with tf.variable_scope(self._name, initializer=tf.orthogonal_initializer()):
            embedded_inputs = tf.nn.embedding_lookup(
                self._embedding_table_source, inputs)                                   # [N, T_in, 256]
            # Encoder
            if prenet_flage:
                prenet_outputs = prenet(embedded_inputs, droupout,
                                        is_training=self.is_training, scope='prenet')   # [N, T_in, 128]
            else:
                prenet_outputs = embedded_inputs

            batch_size = tf.shape(inputs)[0]
            seq_len = tf.tile(tf.expand_dims(tf.shape(inputs)[1], 0), [
                              batch_size], name='sequence_length')

            # BiRnn with the LNGRU
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                LNGRUCell('en_lngru_fw', self._rnn_encoder_dim),
                LNGRUCell('en_lngru_bw', self._rnn_encoder_dim),
                prenet_outputs,
                sequence_length=seq_len,
                dtype=tf.float32)
            encoder_out = tf.concat(outputs, axis=2)

        return encoder_out
