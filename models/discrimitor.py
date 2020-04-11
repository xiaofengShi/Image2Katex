'''
File: discrimitor.py
Project: models
File Created: Thursday, 21st February 2019 3:42:50 pm
Author: xiaofeng (sxf1052566766@163.com)
-----
Last Modified: Friday, 22nd February 2019 10:53:59 am
Modified By: xiaofeng (sxf1052566766@163.com>)
-----
 2018.06 - 2019 Latex Math, Latex Math
'''


from __future__ import absolute_import, division, print_function

import math

import tensorflow as tf
from models.component.trainop import initializer as _initializer
# from wnconv1d import wnconv1d


def TcnConv(incoming, num_filters, filter_size, stride=1, padding='valid',
            data_format='channels_last', dilation_rate=1, activation=None, use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(0, 0.01),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None, bias_regularizer=None, name=None):

    input_shape = incoming.get_shape().as_list()

    if data_format.lower() == 'channels_last':
        channel_axis = -1
    else:
        channel_axis = 1

    if input_shape[channel_axis] is None:
        raise ValueError('The channel dimension of the inputs '
                         'should be defined. Found `None`.')

    input_dim = input_shape[channel_axis]
    # kernel_shape = filter_size + (input_dim, num_filters)

    with tf.variable_scope(name):
        conv_out = tf.layers.conv1d(
            inputs=incoming, filters=num_filters, kernel_size=filter_size, strides=stride,
            padding=padding, data_format=data_format, dilation_rate=dilation_rate,
            use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer)
        return conv_out


class TemporalConvNet(object):
    """ Temporal convulution """

    def __init__(self, num_channels, stride=1, kernel_size=2, dropout=0.5):
        self.kernel_initializer = _initializer('Xavier')
        self.bias_initializer = _initializer('Zero')
        self.kernel_regular = tf.contrib.layers.l1_regularizer(1.0 / 10000)
        self.bias_regular = tf.contrib.layers.l1_regularizer(1.0 / 10000)
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_levels = len(num_channels)
        self.num_channels = num_channels
        self.dropout = dropout
        self.end_points = {}

    def _TemporalBlock(
            self, value, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.5,
            level=0):
        # pad the input sequence
        padded_value1 = tf.pad(value, [[0, 0], [padding, 0], [0, 0]])
        # TCN convoluted
        self.conv1 = TcnConv(
            incoming=padded_value1, num_filters=n_outputs, filter_size=kernel_size, stride=stride,
            dilation_rate=dilation, kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regular,
            bias_regularizer=self.bias_regular, name='conv1d' + str(level) + '_conv1')

        self.end_points[str(level) + '_conv1'] = self.conv1
        # relu and droupout
        self.output1 = tf.nn.dropout(tf.nn.relu(self.conv1), dropout)

        self.end_points[str(level)+'_relu1'] = self.output1
        # pad the sequence
        padded_value2 = tf.pad(self.output1, [[0, 0], [padding, 0], [0, 0]])
        self.conv2 = TcnConv(
            incoming=padded_value2, num_filters=n_outputs, filter_size=kernel_size, stride=stride,
            dilation_rate=dilation, kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regular,
            bias_regularizer=self.bias_regular, name='conv1d' + str(level) + '_conv2')

        self.end_points[str(level)+'_conv2'] = self.conv2

        # relu and droupout
        self.output2 = tf.nn.dropout(tf.nn.relu(self.conv2), dropout)

        self.end_points[str(level)+'_relu2'] = self.output2

        if n_inputs != n_outputs:
            res_x = tf.layers.conv1d(
                inputs=value, filters=n_outputs, kernel_size=1,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                name='layer' + str(level) + '_conv')
        else:
            res_x = value
        resdiual_add = tf.nn.relu(res_x + self.output2)

        self.end_points[str(level) + '_residual'] = res_x

        self.end_points[str(level) + '_residual_add'] = resdiual_add

        return resdiual_add

    def run_model(self, inputs):
        inputs_shape = inputs.get_shape().as_list()
        batch_size = inputs_shape[0]
        conv_out = [inputs]
        for i in range(self.num_levels):
            dilation_size = 2 ** i
            in_channels = inputs_shape[-1] if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            residual_out = self._TemporalBlock(
                value=conv_out[-1],
                n_inputs=in_channels, n_outputs=out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride, dilation=dilation_size,
                padding=(self.kernel_size - 1) * dilation_size,
                dropout=self.dropout, level=i)

            conv_out.append(residual_out)
        # conv_out contains the conv1d out for each leavel
        tcn_out = conv_out[-1]
        # reshape the tcnout to
        tcn_reshaped = tf.reshape(tcn_out, (batch_size, -1))
        # linera out
        linear_out = tf.layers.dense(
            inputs=tcn_reshaped, units=1, activation=tf.sigmoid,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        return conv_out[-1], self.end_points



## 中文OCR任务详解及应用场景
## 中文OCR任务解决方案
## 深度学习技术实现方案综述
## 文本行提取任务算法模型分类及技术选型
## 连续文本提议网络(CTPN)模型原理
## 区域提议网络(RPN)原理分析
## CTPN模型优化目标原理及代码实现
## 不定长文字识别算法模型分类及技术选型
## 卷积循环神经网络(CRNN)模型原理
## 常用卷积网络模型结构
## DenseNet和CTC解码原理及代码实现
## 深度学习的数据标准化原理及选择
## 序列到序列算法模型原理
## 不定长文字识别的编码器设计及代码实现
## 注意力机制原理详解
## 解码器的贪婪搜索和集束搜索选择和实现
## 循环神经网的模型设计策略及代码实现
## 搭建中文OCR任务pipeline
## 技术总结及触类旁通
