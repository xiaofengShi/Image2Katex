'''
Filename: optional.py
Project: tflib
File Created: Sunday, 2nd December 2018 2:27:37 pm
Author: xiaofeng (sxf1052566766@163.com)
--------------------------
Last Modified: Sunday, 2nd December 2018 2:28:25 pm
Modified By: xiaofeng (sxf1052566766@163.com)
---------------------------
: 2018.06 - 2018 . 
'''

import numpy as np
import tensorflow as tf


def initializer(name,  val=0, gain='linear', std=0.01, mean=0.0, range=0.01, alpha=0.01):
    """
    Wrapper function to perform weight initialization using standard techniques
    :parameters:
        name: Name of initialization technique. Follows same names as lasagne.init module
        val: Fill value in case of constant initialization
        gain: one of 'linear','sigmoid','tanh', 'relu' or 'leakyrelu'
        std: standard deviation used for normal / uniform initialization
        mean: mean value used for normal / uniform initialization
        alpha: used when gain = 'leakyrelu'
    """
    if gain is None:
        gain = 1.0
    else:
        if gain in ['linear', 'sigmoid', 'tanh']:
            gain = 1.0
        elif gain == 'leakyrelu':
            gain = np.sqrt(2 / (1 + alpha**2))
        elif gain == 'relu':
            gain = np.sqrt(2)
        else:
            raise NotImplementedError

    if name == 'Constant':
        return tf.constant_initializer(val)
    elif name == 'Zero':
        return tf.zeros_initializer()
    elif name == 'Normal':
        return tf.random_normal_initializer(mean, std)
    elif name == 'Uniform':
        return tf.random_uniform_initializer(-1, 1)
    elif name == 'truncated':
        return tf.truncated_normal_initializer(mean, std)
    elif name == 'GlorotNormal':
        return tf.glorot_normal_initializer(gain)
    elif name == 'HeNormal':
        return tf.initializers.l(gain)
    elif name == 'HeUniform':
        return tf.initializers.he_uniform(gain)
    elif name == 'LecunNormal':
        return tf.initializers.lecun_normal()
    elif name == 'LecunUniform':
        return tf.initializers.lecun_uniform()
    elif name == 'VarianceScaling':
        return tf.variance_scaling_initializer(1.0)
    elif name == 'Orthogonal':
        return tf.orthogonal_initializer(gain)
    elif name == 'Xavier':
        return tf.contrib.layers.xavier_initializer()


# import config as cfg
# 优化器
adadelta_rho = 0.95
adagrad_initial_accumulator_value = 0.1
adam_beta1 = 0.9
adam_beta2 = 0.999
opt_epsilon = 1e-8
ftrl_learning_rate_power = -0.5
ftrl_initial_accumulator_value = 0.1
ftrl_l1 = 0.0
ftrl_l2 = 0.0
momentum = 0.9
rmsprop_momentum = 0.9
rmsprop_decay = 0.9

end_learning_rate = 0.000001
# learning_rate_decay_factor = 0.7
slim = tf.contrib.slim


def configure_optimizer(optimizer_name, learning_rate):
    if optimizer_name == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=adadelta_rho, epsilon=opt_epsilon)
    elif optimizer_name == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate, initial_accumulator_value=adagrad_initial_accumulator_value)
    elif optimizer_name == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate,
                                           epsilon=opt_epsilon)
    elif optimizer_name == 'nadam':
        optimizer = tf.contrib.opt.NadamOptimizer(learning_rate, beta1=0.9,
                                                  beta2=0.999,
                                                  epsilon=1e-08,
                                                  use_locking=False)

    elif optimizer_name == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate,
                                           learning_rate_power=ftrl_learning_rate_power,
                                           initial_accumulator_value=ftrl_initial_accumulator_value,
                                           l1_regularization_strength=ftrl_l1,
                                           l2_regularization_strength=ftrl_l2)
    elif optimizer_name == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=momentum, name='Momentum')
    elif optimizer_name == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate,
                                              decay=rmsprop_decay,
                                              momentum=rmsprop_momentum,
                                              epsilon=opt_epsilon)
    elif optimizer_name == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', optimizer_name)
    return optimizer


def configure_learning_rate(learning_rate_decay_type, learning_rate,
                            decay_steps, learning_rate_decay_rate,
                            global_step):
    """Configures the learning rate.

    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.
    Returns:
      A `Tensor` representing the learning rate.
    """
    # decay_steps = int(
    #     num_samples_per_epoch / flags.batch_size * flags.num_epochs_per_decay)

    if learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(learning_rate,
                                          global_step,
                                          decay_steps,
                                          learning_rate_decay_rate,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif learning_rate_decay_type == 'fixed':
        # return tf.constant(learning_rate, name='fixed_learning_rate')
        return tf.Variable(learning_rate, trainable=False)
    elif learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(learning_rate,
                                         global_step,
                                         decay_steps,
                                         end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         learning_rate_decay_type)
