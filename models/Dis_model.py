'''
File: Net_train_test.py
Project: formula_rocog
File Created: Monday, 2nd July 2018 3:40:09 pm
Author: xiaofeng (sxf1052566766@163.com)
-----
Last Modified: Tuesday, 3rd July 2018 11:58:46 am
Modified By: xiaofeng (sxf1052566766@163.com>)
-----
 2018.06 - 2018 Latex Math, Latex Math
'''
import sys
import time

import numpy as np
import tensorflow as tf
from models.discrimitor import TemporalConvNet
from models.component.trainop import (configure_learning_rate,
                                      configure_optimizer)
from models.model_base import BaseModel


class DisModel(BaseModel):
    def __init__(self, config, vocab, logger, trainable=True):
        super(DisModel, self).__init__(config, vocab, logger)
        self._trainable = trainable
        self.DisGraph = tf.Graph()
        # self.global_step = tf.Variable(0, trainable=False)

    def build_traineval(self):
        with self.DisGraph.as_default():
            self.global_step = tf.Variable(0, trainable=False)
            self.logger.info('Initialize Training model ...')
            self._placeholder()
            self._setup_model()
            self._pipline()
            self._cal_loss()
            self._optimizer()
            self.init_session()
            self._add_summary()
            self.logger.info('Load Train Model Done ...')

    def build_inference(self):
        with self.DisGraph.as_default():
            self.global_step = tf.Variable(0, trainable=False)
            self.logger.info('Initialize inference model ...')
            self._placeholder()
            self._setup_model()
            self._pipline()
            self.init_session()
            self.logger.info('Load Inference Model Done ...')

    def _placeholder(self):
        # self.lr = tf.placeholder(name='leraning_rate', shape=(), dtype=tf.float32)
        self.fake_seq = tf.placeholder(shape=(None, None), dtype=tf.int32, name='fake_seq')
        self.real_seq = tf.placeholder(shape=(None, None), dtype=tf.int32, name='real_seq')
        self.droupout = tf.placeholder(tf.float32, shape=(), name='droupout')

    def _setup_model(self):
        # image encoder
        # out shape is 【batch,H,W,C]
        self.DisModel = TemporalConvNet(num_channels=self._config.model.num_channels)
        self.logger.info('Discrimitor model build done .')

    def _pipline(self):
        # (batch,1)
        self.Disfake, _ = self.DisModel.run_model(inputs=self.fake_seq)
        # (batch,1)
        self.DisReal, _ = self.DisModel.run_model(inputs=self.real_seq)

    def safe_log(self, x, eps=1e-12):
        return tf.log(x + eps)

    def _cal_loss(self):
        # use crossentropy
        if self._config.use_crossentropy:
            error_real = -tf.reduce_mean(self.safe_log(self.DisReal))
            error_fake = -tf.reduce_mean(self.safe_log(1 - self.Disfake))
        else:
            error_real = -tf.reduce_mean(tf.squared_difference(self.DisReal, 0.9))
            error_fake = -tf.reduce_mean(tf.squared_difference(self.Disfake))

        self.loss = error_real + error_fake

        tf.summary.scalar('RealLoss', error_real)
        tf.summary.scalar('FakeLoss', error_fake)
        tf.summary.scalar('TotalLoss', self.loss)

    def init_session(self):
        """Defines self.sess, self.saver and initialize the variables"""
        if self._config.model.gpu_flage:
            _gpu_options = tf.GPUOptions(
                allow_growth=True, per_process_gpu_memory_fraction=self._config.model.get(
                    'gpu_fraction'))
            _DeviceConfig = tf.ConfigProto(device_count={"CPU": 6},
                                           gpu_options=_gpu_options,
                                           intra_op_parallelism_threads=0,
                                           inter_op_parallelism_threads=0)
        else:
            _DeviceConfig = tf.ConfigProto(device_count={"CPU": 6, "GPU": 0},
                                           intra_op_parallelism_threads=0,
                                           inter_op_parallelism_threads=0)
        self.sess = tf.Session(config=_DeviceConfig, graph=self.DisGraph)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=2, write_version=tf.train.SaverDef.V2)
        self.coord = tf.train.Coordinator()
        self.thread = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def _optimizer(self, clip=None):

        self._learning_rate = configure_learning_rate(
            learning_rate_decay_type=self._config.model.learning_type,
            learning_rate=self._config.model.learning_init,
            decay_steps=self._config.model.learning_decay_step,
            learning_rate_decay_rate=self._config.model.learning_decay_rate,
            global_step=self.global_step)

        tf.summary.scalar('lr', self._learning_rate)
        _optimizer_name = self._config.model.get('optimizer').lower()

        if clip is None:
            clip_value = self._config.model.get('clip_value')
        else:
            clip_value = clip

        with tf.variable_scope("train_step"):
            optimizer = configure_optimizer(
                optimizer_name=_optimizer_name, learning_rate=self._learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if clip_value > 0:  # gradient clipping if clip is positive
                    grads, vs = zip(*optimizer.compute_gradients(self.loss))
                    grads, gnorm = tf.clip_by_global_norm(grads, clip_value)
                    self.optimizer = optimizer.apply_gradients(zip(grads, vs), self.global_step)
                else:
                    self.optimizer = optimizer.minimize(self.loss, self.global_step)

    def _run_trainval_epoch(self, train_set, val_set, resotore_iter):
        """Performs an epoch of training

        Args:
            config: Config instance
            train_set: Dataset instance
            val_set: Dataset instance
            epoch: (int) id of the epoch, starting at 0
            lr_schedule: LRSchedule instance that takes care of learning proc

        Returns:
            score: (float) model will select weights that achieve the highest
                score

        """
        # logging
        batch_size = self._config.model.batch_size
        _SaveIter = self._config.model.save_iter
        _DiplayIter = self._config.model.display_iter

        # iterate over dataset
        _train_iters = train_set.generate()

        # _learning = self._learning_rate.eval(session=self.sess)

        # if _learning < 1e-3:
        #     self._learning_rate = tf.convert_to_tensor(1e-3, dtype=tf.float32)

        for idx, (batch_imgs, batch_formulas, batch_formula_length, _) in enumerate(_train_iters):
            # get feed dict
            feed = {self.img: batch_imgs,
                    self.seqs: batch_formulas,
                    self.formula_length: batch_formula_length,
                    self.droupout: self._config.model.get('droupout')
                    }
            # update step
            _, loss_eval, _summary = self.sess.run(
                [self.optimizer, self.loss, self.merged], feed_dict=feed)

            self.SummaryFileWriter.add_summary(
                summary=_summary, global_step=self.global_step.eval(session=self.sess))

            # update learning rate
            resotore_iter += 1
            if resotore_iter % _SaveIter == 0 and resotore_iter != 0:
                self._save_session(resotore_iter)
            if resotore_iter % _DiplayIter == 0:
                _learning = self._learning_rate.eval(session=self.sess)
                msg = 'Current iter [{}] Loss is [{}] and perplexity is [{}] learning rate is [{}]'.format(
                    resotore_iter, loss_eval, np.exp(loss_eval), _learning)
                self.logger.info(msg)
                # if _learning < 1e-3:
                #     self._learning_rate = tf.convert_to_tensor(1e-3, dtype=tf.float32)

        self._save_session(resotore_iter)

        # evaluation
        self.logger.info("Evaluation runing ...")
        scores = self.evaluate(val_set)
        score = scores[self._config.model.metric_val]

        return score, resotore_iter

    def write_prediction(self, test_set):
        """Performs an epoch of evaluation
        Args:
            config: (Config) with batch_size and dir_answers
            test_set:(Dataset) instance
        Returns:
            files: (list) of path to files
            perp: (float) perplexity on test set
        """
        # initialize containers of references and predictions
        if self._config.model.decoding == "greedy":
            refs, hyps = [], [[]]
        elif self._config.model.decoding == "beams_search":
            refs, hyps = [], [[] for i in range(self._config.model.beam_size)]
        names = []
        # iterate over the dataset
        n_words, ce_words = 0, 0  # sum of ce for all words + nb of words
        iters = test_set.generate()
        for idx, (batch_imgs, batch_formulas, batch_formula_length, batch_name) in enumerate(iters):
            names.extend(batch_name)
            feed = {self.img: batch_imgs,
                    self.seqs: batch_formulas,
                    self.formula_length: batch_formula_length,
                    self.droupout: 1.0}
            ce_words_eval, n_words_eval, ids_eval = self.sess.run(
                [self.ce_word, self.n_word, self.logits_valid.ids], feed_dict=feed)

            # TODO(guillaume): move this logic into tf graph
            if self._config.model.decoding == "greedy":
                ids_eval = np.expand_dims(ids_eval, axis=1)

            elif self._config.model.decoding == "beams_search":
                ids_eval = np.transpose(ids_eval, [0, 2, 1])

            n_words += (n_words_eval-2*np.shape(batch_imgs)[0])
            ce_words += ce_words_eval
            for form, preds in zip(batch_formulas, ids_eval):
                refs.append(form[1:-1])
                for i, pred in enumerate(preds):
                    hyps[i].append(pred)
            if idx % 100 == 0 and idx != 0:
                assert len(refs) == len(names)
        # print('n_words,ce_words', n_words, ce_words)

        files = write_answers(refs, hyps, self._vocab.idx_to_token,
                              self._config.model.eval_dir, self._config.dataset.id_end, names)
        self.logger.info('Evaluation write predict done ...')
        perp = - np.exp(ce_words / float(n_words))

        return files, perp

    def _run_evaluate(self, test_set):
        """Performs an epoch of evaluation
        Args:
            test_set: Dataset instance
            params: (dict) with extra params in it
                - "dir_name": (string)
        Returns:
            scores: (dict) scores["acc"] = 0.85 for instance

        """
        files, perp = self.write_prediction(test_set)
        scores = cal_score(files[0], files[1])

        scores["perplexity"] = perp

        return scores

    def predict_batch(self, images):
        if self._config.model.decoding == "greedy":
            hyps = [[]]
        elif self._config.model.decoding == "beams_search":
            hyps = [[] for i in range(self._config.model.beam_size)]

        feed = {self.img: images, self.droupout: 1.0}

        ids_eval = self.sess.run(self.logits_valid.ids, feed_dict=feed)

        if self._config.model.decoding == "greedy":
            ids_eval = np.expand_dims(ids_eval, axis=1)
        elif self._config.model.decoding == "beams_search":
            ids_eval = np.transpose(ids_eval, [0, 2, 1])

        for preds in ids_eval:
            for i, pred in enumerate(preds):
                predict_idx = truncate_end(pred, self._config.dataset.id_end)  # list
                predict_str = u" ".join([self._vocab.idx_to_token[idx] for idx in predict_idx])
                # [['xxxx']] greedy and [[xxx],[xxxx],[xxx],...] beams_search
                hyps[i].append(predict_str)
                # hyps[i].append(predict_idx)

        return hyps

    def evaluate(self, val_set):
        """Evaluates model on test set
        Calls method run_evaluate on test_set and takes care of logging
        Args:
            config: Config
            test_set: instance of class Dataset
        Return:
            scores: (dict) scores["acc"] = 0.85 for instance
        """
        # logging
        _ = self.restore_session()
        # evaluate
        scores = self._run_evaluate(val_set)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                          for k, v in scores.items()])
        self.logger.info("Eval: {}".format(msg))

        return scores

    def trainval(self, train_set, val_set):
        """Global training procedure

        Calls method self.run_epoch and saves weights if score improves.
        All the epoch-logic including the lr_schedule update must be done in
        self.run_epoch
        Args:
            train_set: Dataset instance
            val_set: Dataset instance
            lr_schedule: LRSchedule instance that takes care of learning proc

        Returns:
            best_score: (float)

        """
        best_score = None
        resotore_iter = self.restore_session()
        for epoch in range(self._config.model.n_epochs):
            # logging
            tic = time.time()
            self.logger.info("Epoch {:}/{:}".format(epoch+1, self._config.model.n_epochs))
            # epoch
            score, resotore_iter = self._run_trainval_epoch(train_set, val_set, resotore_iter)
            # save weights if we have new best score on eval
            if best_score is None or score >= best_score:
                best_score = score
                self.logger.info("- New best score ({:04.2f})!".format(best_score))
                self._save_session(resotore_iter)

            # logging
            toc = time.time()
            self.logger.info("- Elapsed time: {:04.2f}".format(toc - tic))
        self.coord.request_stop()
        self.coord.join(self.thread)

        return best_score

    def test(self, test_set):
        _ = self.restore_session()
        _test_iters = test_set.generate()
        preds_ = []
        predict_list = []
        predict_dict = {}
        for idx, (image_data, label) in enumerate(_test_iters):
            import collections
            temp = collections.OrderedDict()
            # remove the start and end flag
            label_str = u" ".join([self._vocab.idx_to_token[idx] for idx in label[1:-1]])
            preds = self.predict_batch([image_data])
            # extract only one element (no batch)
            for hyp in preds:
                preds_.append((hyp[0], label_str))
                temp['predict'] = hyp[0]
                temp['label'] = label_str
                # temp['data'] = image_data
                temp['img_size'] = np.shape(image_data)[:2]
                predict_list.append(temp)
                predict_dict[idx] = temp
            # if idx > 200:
            #     break

        return preds_, predict_list, predict_dict

    def predict_single_img(self, image_data):
        # return is ['xxx','xxxxx']
        preds_ = []
        preds = self.predict_batch([image_data])
        # extract only one element (no batch)
        for hyp in preds:
            preds_.append(hyp[0])

        return preds_
