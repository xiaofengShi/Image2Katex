'''
Filename: encoder.py
Project: models
File Created: Wednesday, 11th July 2018 3:37:08 pm
Author: xiaofeng (sxf1052566766@163.com)
--------------------------
Last Modified: Sunday, 2nd December 2018 4:09:49 pm
Modified By: xiaofeng (sxf1052566766@163.com)
---------------------------
Copyright: 2018.06 - 2018 OnionMath. OnionMath
'''

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from models.component.encoder_cnn import ImageCNN
from models.component.encoder_rnn import RnnEncoder, RnnEncoderOri

""" Encoder section of the seq2seq model """


class Encoder(object):
    """Class with a __call__ method that applies convolutions to an image"""

    def __init__(self, config, trainable):
        self._config = config
        self._name = self._config.model.get('encoder_name')
        self._rnn_encoder_dim = self._config.model.get('rnn_encoder_dim')
        self.is_training = trainable

    def __call__(self, img, cnn_type='densenet'):
        """Applies convolutions to the image

        Args:
            img: batch of img, shape = (batch, height, width, channels), of type
                tf.uint8
            cnn_type: {'cnn','densenet'}

        Returns:
            the encoded images, shape = (batch, h', w', c')

        """
        with tf.variable_scope(self._name):
            if cnn_type == 'densenet':
                from models.component.encoder_densenet import DenseNet
                _image_cnn = DenseNet(x=img, filters_growth=24,
                                      dropout_rate=self._config.model.droupout, filters_out_nums=32,
                                      training=tf.constant(self.is_training, dtype=tf.bool)).cnn_out

            else:
                _image_cnn, end_point = ImageCNN('encoder_cnn', img, self._config)

            # encoder_out = RnnEncoder('encoder_rnn', _image_cnn,
            #                           self._rnn_encoder_dim)
            encoder_out = RnnEncoderOri('encoder_rnn', _image_cnn,
                                        self._rnn_encoder_dim)

        return encoder_out
