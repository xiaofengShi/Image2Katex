'''
Filename: word_embeding.py
Project: models
File Created: Sunday, 2nd December 2018 2:38:45 pm
Author: xiaofeng (sxf1052566766@163.com)
--------------------------
Last Modified: Sunday, 2nd December 2018 2:39:14 pm
Modified By: xiaofeng (sxf1052566766@163.com)
---------------------------
: 2018.06 - 2018 . 
'''

import tensorflow as tf


def Embedding(name, vocab_embeding, input_sequence):
    """
    Creates an embedding matrix of dimensions n_symbols x embeding_dim upon first use.
    Looks up embedding vector for each input symbol

    :parameters:
        name: name of embedding matrix tensor variable
        vocab_embeding: full vocabulary embedings
        input_sequence: input symbols tensor
    return:
        word embeding for current input_sequence tensor 
        shape is (batch_size,1+input_sequence_length,embeding_dim)
    """

    with tf.device("/cpu:0"), tf.variable_scope(name):
        label_embeding = tf.nn.embedding_lookup(vocab_embeding, input_sequence)
        return label_embeding


def embedding_initializer():
    """Returns initializer for embeddings"""
    def _initializer(shape, dtype, partition_info=None):
        _Em = tf.random_uniform(shape, minval=-1.0, maxval=1.0, dtype=dtype)
        # _init_func = tf.orthogonal_initializer(dtype=dtype)
        _Em = tf.nn.l2_normalize(_Em, -1)
        return _Em

    return _initializer
