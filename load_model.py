'''
File: load_model.py
Project: image2katex
File Created: Wednesday, 26th December 2018 12:40:54 pm
Author: xiaofeng (sxf1052566766@163.com)
-----
Last Modified: Wednesday, 26th December 2018 12:45:11 pm
Modified By: xiaofeng (sxf1052566766@163.com>)
-----
Copyright 2018.06 - 2018 onion Math, onion Math
'''


from __future__ import print_function

import argparse
import collections
import os
import shutil
import sys
from pprint import pprint

import numpy as np
from PIL import Image

import config as cfg
import init_logger
from models.seq2seq_model import Seq2SeqAttModel
from utils.TextUtil import simplify
from utils.general import get_img_list, run
from utils.process_image import (TIMEOUT, crop_image, generate_image_data,
                                 image_process, padding_img, resize_img)
from utils.render_image import latex_to_image
from utils.util import render_to_html

""" Load model for the webserver """


class LoadModel(object):
    def __init__(self, ConfClass, _config, _vocab, logger, trainable=False):
        self.ConfClass=ConfClass
        self.config = _config
        self.vocab = _vocab
        self.trainable = trainable
        self.logger = logger
        self.target_height = self.vocab.target_height
        self.bucket_size = self.vocab.bucket_size
        self.setup()

    def setup(self):
        """ Load model """
        self.Model = Seq2SeqAttModel(config=self.config, vocab=self.vocab,
                                     logger=self.logger, trainable=self.trainable)
        self.Model.build_inference()
        _ = self.Model.restore_session()

        self.temp_path = os.path.abspath(self.config.predict.temp_path)
        self.preprocess_dir = os.path.abspath(self.config.predict.preprocess_dir)
        self.render_path = os.path.abspath(self.config.predict.render_path)

        self.ConfClass.create_dir(
            [self.temp_path, self.preprocess_dir, self.render_path])

    def run_im2latex(self, image_path):
        """ if there is no predict_image given, then input the image_path"""

        image_path = os.path.abspath(image_path)
        assert os.path.exists(image_path), 'Error, the {:s} not exist'.format(image_path)
        assert os.path.isfile(image_path), 'Input {image_path} must be file'.format(image_path)

        root, img_name = os.path.split(image_path)

        if img_name.split('.')[-1] == 'pdf':
            convert_cmd = "magick convert -density {} -quality {} {} {}".format(
                200, 100, os.path.join(root, img_name), os.path.join(
                    self.preprocess_dir, "{}.png".format(img_name.split('.')[0])))
            run(cmd=convert_cmd, timeout_sec=TIMEOUT)
        else:
            src_dir = image_path
            dst_dir = os.path.join(self.preprocess_dir, img_name)
            shutil.copy(src=src_dir, dst=dst_dir)
        # preprocess the image
        _img_name_no_ext = img_name.split('.')[0]
        # process image to the uniform style
        # binary,adaptive-filter,find-existed-pixel and crop it
        # resize image based targetd given by the training dataset
        image_process_flage = image_process(
            input_dir=self.preprocess_dir, preprocess_dir=self.preprocess_dir,
            render_out=self.render_path, file_name=img_name, target_height=self.target_height,
            bucket_size=self.bucket_size, _logger=self.logger)
        if not image_process_flage:
            pass
        # convert processed image to the numpy data
        image_data = generate_image_data(os.path.join(
            self.preprocess_dir, _img_name_no_ext + '.png'), self.logger, False)

        # predict the image based image data
        _predict_latex_list = self.Model.predict_single_img(image_data)
        _LatexWant = _predict_latex_list[0]
        # get the directory for the pwd file
        pwd = os.path.abspath(os.getcwd())
        # switch the directory to the render path
        if self.render_path not in pwd:
            os.chdir(self.render_path)
        render_flag = latex_to_image(_LatexWant, _img_name_no_ext, self.logger)
        # switch directory to the pwd
        os.chdir(pwd)
        if render_flag:
            param_croped = (
                os.path.join(self.render_path, _img_name_no_ext+'.png'),
                self.render_path, _img_name_no_ext+'.png', self.logger)
            _ = crop_image(param_croped)
        temp = collections.OrderedDict()
        temp['process_img'] = _img_name_no_ext+'.png'
        temp['predict_latex'] = _LatexWant
        temp['render_dir'] = _img_name_no_ext + '.png' if os.path.exists(
            os.path.join(self.render_path, _img_name_no_ext + '.png')) else None

        return temp


# if __name__ == "__main__":
#     while True:
#         p = LoadModel()
#         p.run_im2latex()
