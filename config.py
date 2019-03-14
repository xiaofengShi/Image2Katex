'''
Filename: config.py
Project: image2katex
File Created: Wednesday, 5th December 2018 5:32:00 pm
Author: xiaofeng (sxf1052566766@163.com)
--------------------------
Last Modified: Wednesday, 5th December 2018 5:32:04 pm
Modified By: xiaofeng (sxf1052566766@163.com)
---------------------------
Copyright: 2018.06 - 2018 OnionMath. OnionMath
'''

import logging
import os
from math import ceil

import dominate
import numpy as np
from dominate.tags import *
from easydict import EasyDict as edict
import yaml
from utils import util


"""
 Usage:
    cfg=Config().config

    cfg.dataset.id_end
    cfg.dataset.get('id_end')
 """


class ConfigSeq2Seq:
    def __init__(self, data_type, gpu, encoder_type='conv'):
        self._data_type = data_type
        self.gpu = gpu
        self.encoder_type = encoder_type
        self._configs = edict()
        self._configs.datatype = self.encoder_type
        self.model()
        self.dataset()
        self.predict()
        self.initalize_dirs()

    def model(self):
        _model = edict()
        _model.batch_size = 16
        _model.test_batch_size = 1
        _model.gpu_flage = self.gpu >= 0
        _model.gpu_fraction = 0.7
        # _model.optimizer = 'momentum'
        # _model.learning_type = 'exponential'  # ['exponential','fixed','polynomial']
        _model.learning_decay_step = 8000
        _model.learning_decay_rate = 0.94
        _model.encoder_name = 'Encode'
        # The different between Augment and conv:
        #  Augment: image enhance and adadalta optimizer
        #  conv: image normal and momentum optmizer
        if self.encoder_type == 'Augment':
            # _model.encoder_type = 'Augment'  # ['Augment','conv']
            # _model.learning_init = 0.1
            # _model.optimizer = 'adadelta'
            # _model.learning_type = 'fixed'  # ['exponential','fixed','polynomial']
            _model.encoder_type = 'Augment'
            _model.learning_init = 0.001
            _model.optimizer = 'momentum'
            _model.learning_type = 'exponential'  # ['exponential','fixed','polynomial']
        else:
            _model.encoder_type = 'conv'
            _model.learning_init = 0.001
            _model.optimizer = 'momentum'
            _model.learning_type = 'exponential'  # ['exponential','fixed','polynomial']
        _model.decoder_name = 'DecoderAtt'
        _model.encoder_cnn = "vanilla"
        _model.droupout = 0.3  # droupout rate
        _model.positional_embeddings = True
        _model.rnn_encoder_dim = 256  # rnn encoder num unit
        _model.embeding_dims = 80  # word embeding dimision
        _model.rnn_decoder_dim = 512  # rnn decoder num unit
        _model.att_dim = 512
        _model.clip_value = 5
        _model.save_iter = 500
        _model.display_iter = 100
        _model.beam_size = 5
        _model.div_gamma = 1
        _model.div_prob = 0
        _model.n_epochs = 1000
        _model.MaxPredictLength = 200
        _model.decoding = 'beams_search'  # chose from ['greedy','beams_search']
        # _model.decoding='greedy'
        _model.metric_val = 'perplexity'
        if self._data_type == 'handwritten':
            _model.model_saved = '/home/xiaofeng/data/image2latex/handwritten/model_saved/' + _model.encoder_type
        elif self._data_type == 'original':
            _model.model_saved = '/home/xiaofeng/data/image2latex/original/model_saved/' + _model.encoder_type
        else:
            _model.model_saved = '/home/xiaofeng/data/image2latex/merged/model_saved/' + _model.encoder_type
        _model.ckpt_name = 'seq2seqAtt'
        _model.ckpt_dir = os.path.abspath(os.path.join(_model.model_saved, 'ckpt'))
        _model.eval_dir = os.path.abspath(os.path.join(_model.model_saved,  'eval'))
        _model.summary_dir = os.path.abspath(os.path.join(_model.model_saved,  'summary'))

        _model.log_dir = os.path.abspath('./log')  # log path
        _model.log_name = 'Im2Katex'
        _model.log_file_name = 'Im2Katex.log'
        self._configs.model = _model

    def dataset(self):
        _dataset = edict()
        _dataset.id_start = 0
        _dataset.id_end = 1
        _dataset.id_unk = 2
        _dataset.id_pad = 3
        if self._data_type == 'handwritten':
            _dataset.image_folder = [
                '/home/xiaofeng/data/image2latex/handwritten/process/img_padding']
            _dataset.prepared_folder = ['./data/im2latex_dataset/handwritten/prepared/']
            _dataset.vocabulary_file = _dataset.prepared_folder[0] + 'properties.npy'
        elif self._data_type == 'original':
            _dataset.image_folder = ['/home/xiaofeng/data/image2latex/original/process/img_padding']
            _dataset.prepared_folder = ['./data/im2latex_dataset/original/prepared/']
            _dataset.vocabulary_file = _dataset.prepared_folder[0] + 'properties.npy'
        else:
            _dataset.image_folder = [
                '/home/xiaofeng/data/image2latex/handwritten/process/img_padding',
                '/home/xiaofeng/data/image2latex/original/process/img_padding']
            _dataset.prepared_folder = [
                './data/im2latex_dataset/merged/prepared/handwritten/',
                './data/im2latex_dataset/merged/prepared/original/']
            # The properties are the same for the handwritten and original dataset
            # just use one is ok
            _dataset.vocabulary_file = _dataset.prepared_folder[0] + 'properties.npy'

        self._configs.dataset = _dataset

    def predict(self):
        """ 
        The predict details want to be displayed o web, 
        so the root image is the "static" which is the flask defaulet static folder 
        """
        _predict = edict()
        _predict.web_path = './templates'
        # root dir
        _predict.temp_path = './static'
        # preprocess folder for the predict
        _predict.preprocess_dir = os.path.join(_predict.temp_path, 'preprocess')
        # save details on the numpy format
        _predict.npy_path = os.path.join(_predict.temp_path, 'npy')
        # # if the input is an pdf, the convert it
        # _predict.pdf_path = os.path.join(_predict.preprocess, 'pdf')
        # # crop the input image
        # _predict.croped_path = os.path.join(_predict.preprocess, 'croped')
        # # resize the input image
        # _predict.resized_path = os.path.join(_predict.preprocess, 'resized')
        # # pad the input image
        # _predict.pad_path = os.path.join(_predict.preprocess, 'pad')
        # render the image based on latex predicted by the given image
        _predict.render_path = os.path.join(_predict.temp_path, 'render')
        # # crop the rendered image and save it
        # _predict.render_out_path = os.path.join(_predict.temp_path, 'render', 'out')

        self._configs.predict = _predict

    def create_dir(self, dirs):
        assert type(dirs) == list, 'Input dir must be a list type '
        for cur_dir in dirs:
            if not os.path.exists(cur_dir):
                os.makedirs(cur_dir)

    def initalize_dirs(self):
        self.create_dir([self._configs.model.model_saved, self._configs.model.ckpt_dir,
                         self._configs.model.eval_dir, self._configs.model.summary_dir, self._configs.model.log_dir])

    def save_cfg(self):
        config_file = os.path.join(os.getcwd(), 'Im2Katex_config.yml')
        with open(config_file, 'w') as outfile:
            yaml.dump(self._configs, outfile, default_flow_style=False)


class VocabSeq2Seq:
    def __init__(self, config, logger):
        self._config = config
        self._logger = logger
        self.load_vocab()

    def load_vocab(self):
        vocab_dir = os.path.abspath(self._config.dataset.vocabulary_file)
        vocabulary = np.load(vocab_dir).tolist()
        self.vocab_size = vocabulary['vocab_size']
        self.idx_to_token = vocabulary['idx_to_str']
        self.token_to_idx = vocabulary['str_to_idx']
        self.bucket_size = [(687, 24), (598, 24), (597, 32), (450, 32),
                            (569, 64), (762, 48), (703, 64), (256, 32),
                            (591, 40), (525, 40), (335, 40), (593, 48), (152, 48),
                            (505, 64), (311, 64), (381, 32), (197, 32), (398, 40),
                            (83, 40), (376, 64), (245, 64), (199, 24), (738, 40),
                            (140, 32), (678, 32), (676, 48), (441, 64), (351, 24),
                            (636, 64), (126, 24), (147, 40), (777, 24), (512, 24),
                            (512, 48), (660, 40), (218, 48), (359, 48), (778, 64),
                            (461, 40), (274, 24), (272, 40), (287, 48), (317, 32),
                            (210, 40), (522, 32), (178, 64), (430, 24), (434, 48)]
        self.target_height = list(set(idx[1] for idx in self.bucket_size))
        self._logger.info('Vocab size is [{:d}]'.format(self.vocab_size))


# save image to the disk

def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = im_data
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        if aspect_ratio < 1.0:
            im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
        util.save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)

    webpage.add_images(ims, txts, links, width=width)


class HTML:
    def __init__(self, web_dir, title, reflesh=0):
        self.title = title
        self.web_dir = web_dir
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)

        self.doc = dominate.document(title=title)
        self.headers = ['idx', 'image', 'rendered', 'latex']

        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh))

        self.idx = 0
        self.pred = 0
        self.add_header('Display')
        self.add_table()

    def add_header(self, str):
        with self.doc:
            h2(str)

    def add_end(self, xxx):
        with self.doc:
            self.header = h3
            self.header(xxx)

    def add_table(self, border=1):

        self.t = table(border=border, style="table-layout: fixed;")

        with self.t:
            with tr(style="word-wrap: break-word;", halign="center", valign="top"):
                for header in self.headers:
                    with th():
                        with p():
                            p(header)

        self.doc.add(self.t)

    def add_images(self, ims, txts, links, width=400):
        with self.t:
            for im_path, latex, render_path in zip(ims, txts, links):
                self.idx += 1
                with tr(style="word-wrap: break-word;", halign="center", valign="top"):
                    with td():
                        with p():
                            num_str = str(self.idx)
                            p(num_str)
                    with td():
                        img(style="width:%dpx" % width, src=im_path)
                    with td():
                        if render_path is not None:
                            img(style="width:%dpx" % width, src=render_path)
                            self.pred += 1

                        else:
                            with p():
                                p('None')
                    with td():
                        with p():
                            p(latex)

        perp = float(self.pred / self.idx)
        xxx = 'Iter is {}  and ACC is {}'.format(self.idx, perp)
        self.add_end(xxx)

    def save(self):
        html_file = '%s/predict_200.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


class ConfigServer:
    batch_size = 16
    test_batch_size = 1
    gpu_fraction = 0.48
    optimizer = 'momentum'
    learning_type = 'exponential'  # ['exponential','fixed','polynomial']
    learning_init = 0.1
    learning_decay_step = 8000
    learning_decay_rate = 0.94
    encoder_name = 'Encode'
    encoder_type = 'conv'   # ['conv_lngru','conv']
    decoder_name = 'DecoderAtt'
    encoder_cnn = "vanilla"
    droupout = 0.3  # droupout rate
    positional_embeddings = True
    rnn_encoder_dim = 256  # rnn encoder num unit
    embeding_dims = 80  # word embeding dimision
    rnn_decoder_dim = 512  # rnn decoder num unit
    att_dim = 512
    clip_value = 5
    save_iter = 500
    display_iter = 100
    beam_size = 5
    div_gamma = 1
    div_prob = 0
    n_epochs = 1000
    MaxPredictLength = 200
    decoding = 'beams_search'  # chose from ['greedy','beams_search']
    metric_val = 'perplexity'


class ConfigErrorChecker:
    def __init__(self, gpu):
        self._configs = edict()
        self.gpu = gpu
        self.model()
        self.dataset()
        self.predict()
        self.initalize_dirs()

    def model(self):
        _model = edict()
        _model.batch_size = 32
        _model.test_batch_size = 1
        _model.gpu_fraction = 0.48
        _model.gpu_flage = self.gpu >= 0
        _model.optimizer = 'momentum'
        _model.learning_type = 'exponential'  # ['exponential','fixed','polynomial']
        _model.learning_init = 0.1
        _model.learning_decay_step = 10000
        _model.learning_decay_rate = 0.96
        _model.errche_encoder_name = 'Encode_errche'
        _model.errche_encoder_type = 'Prenet'   # ['Prenet','uniform']
        _model.errche_decoder_name = 'DecoderAtt_errche'
        _model.droupout = 0.3  # droupout rate
        _model.errche_rnn_encoder_dim = 128  # rnn encoder num unit
        _model.errche_embeding_dims_target = 128  # word embeding dimision for source
        _model.errche_embeding_dims_source = 128  # word embeding dimision for target
        _model.errche_rnn_decoder_dim = 256  # rnn decoder num unit
        _model.att_dim = 256
        _model.clip_value = 5
        _model.save_iter = 500
        _model.display_iter = 100
        _model.beam_size = 5
        _model.div_gamma = 1
        _model.div_prob = 0
        _model.n_epochs = 1000
        _model.MaxPredictLength = 200
        _model.decoding = 'greedy'  # chose from ['greedy','beams_search']
        _model.metric_val = 'perplexity'
        _model.model_saved = '/home/xiaofeng/data/image2latex/ErrorCheck/model_saved/' + _model.errche_encoder_type
        _model.ckpt_name = 'ErrorCheck'
        _model.ckpt_dir = os.path.abspath(os.path.join(_model.model_saved, 'ckpt'))
        _model.eval_dir = os.path.abspath(os.path.join(_model.model_saved,  'eval'))
        _model.summary_dir = os.path.abspath(os.path.join(_model.model_saved,  'summary'))
        _model.log_dir = os.path.abspath('./log')  # log path
        _model.log_name = 'ErrorChecker'
        _model.log_file_name = 'ErrorChecker.log'
        self._configs.model = _model

    def dataset(self):
        _dataset = edict()
        _dataset.id_start = 0
        _dataset.id_end = 1
        _dataset.id_unk = 2
        _dataset.id_pad = 3
        _dataset.prepared_folder = ['./data/errorchecker_dataset/prepared']
        _dataset.vocabulary_file = './data/errorchecker_dataset/prepared/properties.npy'
        _dataset.bucket_size = [(117, 121), (84, 86), (42, 42), (61, 62), (52, 52),
                                (74, 18), (72, 73), (179, 192), (143, 147), (198, 58), (33, 33),
                                (22, 21), (99, 101)]
        self._configs.dataset = _dataset

    def predict(self):
        """ 
        The predict details want to be displayed o web, 
        so the root image is the "static" which is the flask defaulet static folder 
        """
        _predict = edict()
        _predict.web_path = './templates'
        # root dir
        _predict.temp_path = './static'
        # preprocess folder for the predict
        _predict.preprocess_dir = os.path.join(_predict.temp_path, 'preprocess')
        # save details on the numpy format
        _predict.npy_path = os.path.join(_predict.temp_path, 'npy')
        # # if the input is an pdf, the convert it
        # _predict.pdf_path = os.path.join(_predict.preprocess, 'pdf')
        # # crop the input image
        # _predict.croped_path = os.path.join(_predict.preprocess, 'croped')
        # # resize the input image
        # _predict.resized_path = os.path.join(_predict.preprocess, 'resized')
        # # pad the input image
        # _predict.pad_path = os.path.join(_predict.preprocess, 'pad')
        # render the image based on latex predicted by the given image
        _predict.render_path = os.path.join(_predict.temp_path, 'render')
        # # crop the rendered image and save it
        # _predict.render_out_path = os.path.join(_predict.temp_path, 'render', 'out')

        self._configs.predict = _predict

    def create_dir(self, dirs):
        assert type(dirs) == list, 'Input dir must be a list type '
        for cur_dir in dirs:
            if not os.path.exists(cur_dir):
                os.makedirs(cur_dir)

    def initalize_dirs(self):
        self.create_dir([self._configs.model.model_saved, self._configs.model.ckpt_dir,
                         self._configs.model.eval_dir, self._configs.model.summary_dir, self._configs.model.log_dir])

    def save_cfg(self):
        config_file = os.path.join(os.getcwd(), 'Errorchcker_config.yml')
        with open(config_file, 'w') as outfile:
            yaml.dump(self._configs, outfile, default_flow_style=False)


class VocabErrorChecker:
    def __init__(self, config, logger):
        self._config = config
        self._logger = logger
        self.START_ID = 0
        self.EOS_ID = 1
        self.UNK_ID = 2
        self.PAD_ID = 3
        self.load_vocab()

    def load_vocab(self):
        vocab_dir = os.path.abspath(self._config.dataset.vocabulary_file)
        vocabulary = np.load(vocab_dir).tolist()
        self.vocab_size = vocabulary['vocab_size']
        self.idx_to_token = vocabulary['idx_to_str']
        self.token_to_idx = vocabulary['str_to_idx']
        self.bucket_size = [(117, 121), (84, 86), (42, 42), (61, 62), (52, 52),
                            (74, 18), (72, 73), (179, 192), (143, 147), (198, 58), (33, 33),
                            (22, 21), (99, 101)]
        self.target_height = list(set(idx[1] for idx in self.bucket_size))

        self.errche_vocab_size_source = self.vocab_size  # do not contain the
        self.errche_vocab_size_targ = self.vocab_size
        self._logger.info('Vocab size is [{:d}]'.format(self.vocab_size))


class ConfigDis:
    def __init__(self, gpu):
        self._configs = edict()
        self.gpu = gpu
        self.model()
        self.dataset()
        self.predict()
        self.initalize_dirs()

    def model(self):
        _model = edict()
        _model.gpu_flage = self.gpu >= 0
        _model.gpu_fraction = 0.7
        _model.batch_size = 32
        _model.embeding_dims = 80
        _model.hidden_units = 450
        _model.levels = 3
        _model.num_channels = [_model.hidden_units] * (_model.levels - 1) + [_model.embeding_dims]
        _model.optimizer = 'momentum'
        _model.learning_type = 'exponential'  # ['exponential','fixed','polynomial']
        _model.clip_value = 0.15
        _model.learning_decay_step = 8000
        _model.learning_decay_rate = 0.94
        _model.learning_init = 0.001
        _model.save_iter = 500
        _model.display_iter = 100
        _model.n_epochs = 1000
        _model.MaxPredictLength = 200
        _model.dis_model = 'TCN'
        _model.use_crossentropy=True
        _model.metric_val = 'perplexity'
        _model.model_saved = '/home/xiaofeng/data/image2latex/Discri/model_saved/' + _model.dis_model
        _model.ckpt_name = 'DisModel'
        _model.ckpt_dir = os.path.abspath(os.path.join(_model.model_saved, 'ckpt'))
        _model.eval_dir = os.path.abspath(os.path.join(_model.model_saved,  'eval'))
        _model.summary_dir = os.path.abspath(os.path.join(_model.model_saved,  'summary'))
        _model.log_dir = os.path.abspath('./log')  # log path
        _model.log_name = 'DisModel'
        _model.log_file_name = 'DisModel.log'
        self._configs.model = _model

    def dataset(self):
        _dataset = edict()
        _dataset.id_start = 0
        _dataset.id_end = 1
        _dataset.id_unk = 2
        _dataset.id_pad = 3
        _dataset.prepared_folder = ['./data/errorchecker_dataset/prepared']
        _dataset.vocabulary_file = './data/errorchecker_dataset/prepared/properties.npy'
        _dataset.bucket_size = [(117, 121), (84, 86), (42, 42), (61, 62), (52, 52),
                                (74, 18), (72, 73), (179, 192), (143, 147), (198, 58), (33, 33),
                                (22, 21), (99, 101)]
        self._configs.dataset = _dataset
