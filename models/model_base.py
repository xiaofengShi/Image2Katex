'''
Filename: model_base.py
Project: models
File Created: Friday, 30th November 2018 2:55:28 pm
Author: xiaofeng (sxf1052566766@163.com)
--------------------------
Last Modified: Thursday, 6th December 2018 7:40:45 pm
Modified By: xiaofeng (sxf1052566766@163.com)
---------------------------
: 2018.06 - 2018 . 
'''

import codecs
import os
import sys

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


class BaseModel(object):
    """Generic class for tf models"""

    def __init__(self, config, vocab, logger):
        """Defines self._config
        """
        self._config = config
        self._vocab = vocab
        self.logger = logger
        self._ckpt_dir = self._config.model.get('ckpt_dir')
        self._summary_dir = self._config.model.get('summary_dir')
        self._ModelName = self._config.model.get("ckpt_name")
        self._SaveIter = self._config.model.get("save_iter")
        self._DiplayIter = self._config.model.get("display_iter")
        # tf.reset_default_graph()  # saveguard if previous model was defined

    def how_many_paras(self, checkpoint_path):
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        # for key in var_to_shape_map:
        #     print("tensor_name: ", key)
        #     print(reader.get_tensor(key))
        # print('how many paras in the ckpt:', len(var_to_shape_map))
        return len(var_to_shape_map)

    def init_session(self):
        NotImplemented

    def restore_session(self,pretrainde=None):
        """Reload weights into session

        Args:
            sess: tf.Session()
        return:
            restore_iter: int 
        """
        if pretrainde is None:
            dir_model = self._ckpt_dir
        else:
            dir_model=pretrainde
        # print('Model directionary is [{:s}]'.format(dir_model))
        restore_iter = 0
        if os.listdir(dir_model):
            ckpt = tf.train.get_checkpoint_state(dir_model)
            self.logger.info(
                "Reloading the latest trained model, the name is {}".format(
                    ckpt.model_checkpoint_path))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

            stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[0]
            restore_iter = int(stem.split('_')[-1])
            self.sess.run(self.global_step.assign(restore_iter))
            self.logger.info('Reloading pretrained weights done ...')

        return restore_iter

    def _save_session(self, iters):
        """Saves session"""
        # logging
        sys.stdout.write("\r- Saving model...")
        sys.stdout.flush()

        filename = self._ModelName+'_{:d}'.format(iters) + '.ckpt'
        file_dir = os.path.join(self._ckpt_dir, filename)
        self.saver.save(self.sess, file_dir)

        calculate_params = False
        if calculate_params:
            msg = 'Restore the weight files form: {}'.format(file_dir)
            # check how many paras in the ckpt
            para_nuims = self.how_many_paras(file_dir)
            self.logger.info('Para num is [{}]'.format(para_nuims))
        # logging
        sys.stdout.write("\r")
        sys.stdout.flush()
        self.logger.info('Current iter is [{:d}]'.format(iters))
        self.logger.info("Saved model in {}".format(self._ckpt_dir))

    def close_session(self):
        """Closes the session"""
        self.sess.close()

    def _add_summary(self):
        """Defines variables for Tensorboard

        Args:
            dir_output: (string) where the results are written

        """
        self.merged = tf.summary.merge_all()
        self.SummaryFileWriter = tf.summary.FileWriter(logdir=self._summary_dir, flush_secs=2,
                                                       graph=self.sess.graph)

    def convert_str(self, strs):
        return codecs.escape_decode(bytes(strs, 'utf-8'))[0].decode('utf-8')
