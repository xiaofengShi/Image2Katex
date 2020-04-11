'''
File: run.py
Project: image2katex
File Created: Saturday, 29th December 2018 6:35:25 pm
Author: xiaofeng (sxf1052566766@163.com)
-----
Last Modified: Saturday, 29th December 2018 6:35:57 pm
Modified By: xiaofeng (sxf1052566766@163.com>)
-----
 2018.06 - 2018 Latex Math, Latex Math
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
from dataset_iter import DataIteratorSeq2SeqAtt, DataIteratorErrorChecker
from models.seq2seq_model import Seq2SeqAttModel
from models.error_checker_model import ErrorCheckerModel

from utils.general import get_img_list, run
from utils.TextUtil import simplify
from utils.process_image import (TIMEOUT, crop_image, generate_image_data,
                                 image_process, padding_img, resize_img)
from utils.render_image import latex_to_image
from utils.util import render_to_html


def im2katex(parameters):
    _dataset_type = parameters.data_type
    _gpu = parameters.gpu
    _encoder_type = parameters.encoder_type
    _Configure = cfg.ConfigSeq2Seq(_dataset_type, _gpu, _encoder_type)
    # save the configure as the yaml format
    _Configure.save_cfg()
    # Get configures for the project
    _config = _Configure._configs
    # pprint the configure
    pprint(_config)
    logger = init_logger.get_logger(
        _loggerDir=_config.model.log_dir, log_path=_config.model.log_file_name,
        logger_name=_config.model.log_name)

    logger.info('Logging is working ...')
    # Generate the vocab
    _vocab = cfg.VocabSeq2Seq(_config, logger)

    if parameters.mode == 'trainval':
        logger.info('This is an trainval mode')
        _TrainDataLoader = DataIteratorSeq2SeqAtt(_config, logger, ['train'])
        _ValDataLoader = DataIteratorSeq2SeqAtt(_config, logger, ['validate'])
        Model = Seq2SeqAttModel(config=_config, vocab=_vocab, logger=logger,
                                trainable=True)
        Model.build_traineval()
        Model.trainval(_TrainDataLoader, _ValDataLoader)

    elif parameters.mode == 'val':
        """ Validate the dataset """
        logger.info('This is an validate mode')
        # _ValDataLoader = DataIteratorSeq2SeqAtt(_config, logger, ['train', 'validate', 'test'])
        _ValDataLoader = DataIteratorSeq2SeqAtt(_config, logger, ['validate'])
        Model = Seq2SeqAttModel(config=_config, vocab=_vocab, logger=logger,
                                trainable=False)
        Model.build_traineval()
        scores = Model.evaluate(_ValDataLoader)
        print('evaluation score is:', scores)

    elif parameters.mode == 'test':
        """ run the test dataset """
        logger.info('This is an test model')
        _TestDataLoader = DataIteratorSeq2SeqAtt(_config, logger, ['test'])

        Model = Seq2SeqAttModel(config=_config, vocab=_vocab, logger=logger, trainable=False)
        Model.build_inference()
        _, predict_npy, _ = Model.test(_TestDataLoader)

        np.save('./temp/predict/predict_out', predict_npy)
    else:
        logger.info(
            'This is an predict model, you can predict a sigle image predict files under a given directory')
        temp_path = os.path.abspath(_config.predict.temp_path)
        npy_path = os.path.abspath(_config.predict.npy_path)
        preprocess_dir = os.path.abspath(_config.predict.preprocess_dir)
        render_path = os.path.abspath(_config.predict.render_path)
        webpath = os.path.abspath(_config.predict.web_path)

        _Configure.create_dir(
            [temp_path, npy_path, preprocess_dir, render_path, webpath])

        webpage = cfg.s(webpath, title='Predict_Image_Display')
        target_height = _vocab.target_height
        bucket_size = _vocab.bucket_size
        Model = Seq2SeqAttModel(config=_config, vocab=_vocab, logger=logger, trainable=False)
        Model.build_inference()
        _ = Model.restore_session()

        while True:
            """ Load the model and restore the weights """

            """ if there is no predict_image given, then input the image_path"""
            if parameters.predict_image:
                image_path = parameters.predict_image
            else:
                # python3
                image_path = input("Input file or directory want to be predict ... >")
                # delete the whitespace
                image_path = simplify(text=image_path)
                if image_path.lower() == 'exit':
                    break
            if os.path.isdir(image_path):
                image_list = get_img_list(image_path)
                logger.info('Total image num want to predict {}'.format(len(image_list)))
            elif os.path.isfile(image_path):
                root, name = os.path.split(image_path)
                image_list = [name]
                image_path = root
            else:
                logger.warn('Please check the input image path')
                continue

            if not image_list:
                logger.warn('Image list is None')
                continue

            predict_details = list()
            image_nums = len(image_list)
            for idx, img_name in enumerate(image_list):
                if img_name.split('.')[-1] == 'pdf':
                    convert_cmd = "magick convert -density {} -quality {} {} {}".format(
                        200, 100, os.path.join(image_path, img_name), os.path.join(
                            preprocess_dir, "{}.png".format(img_name.split('.')[0])))
                    run(cmd=convert_cmd, timeout_sec=TIMEOUT)
                else:
                    src_dir = os.path.join(image_path, img_name)
                    dst_dir = os.path.join(preprocess_dir, img_name)
                    shutil.copy(src=src_dir, dst=dst_dir)
                # preprocess the image
                _img_name_no_ext = img_name.split('.')[0]
                image_process_flage = image_process(
                    input_dir=preprocess_dir, preprocess_dir=preprocess_dir,
                    render_out=render_path, file_name=img_name, target_height=target_height,
                    bucket_size=bucket_size, _logger=logger)

                if not image_process_flage:
                    continue
                image_data = generate_image_data(os.path.join(
                    preprocess_dir, _img_name_no_ext+'.png'), logger, False)
                # predict the image based image data
                _predict_latex_list = Model.predict_single_img(image_data)
                _LatexWant = _predict_latex_list[0]
                # get the directory for the pwd file
                pwd = os.path.abspath(os.getcwd())
                # switch the directory to the render path
                if render_path not in pwd:
                    os.chdir(render_path)
                render_flag = latex_to_image(_LatexWant, _img_name_no_ext, logger)
                # switch directory to the pwd
                os.chdir(pwd)
                if render_flag:
                    param_croped = (
                        os.path.join(render_path, _img_name_no_ext+'.png'),
                        render_path, _img_name_no_ext+'.png', logger)
                    _ = crop_image(param_croped)
                temp = collections.OrderedDict()
                temp['input_dir'] = os.path.join('preprocess', _img_name_no_ext+'.png')
                temp['predict_latex'] = _LatexWant
                temp['render_dir'] = os.path.join(
                    'render', _img_name_no_ext + '.png') if os.path.exists(
                    os.path.join(render_path, _img_name_no_ext + '.png')) else None
                predict_details.append(temp)

                if idx % 200 == 0 and idx != 0:
                    render_to_html(webpage=webpage, predict_details=predict_details,
                                   npy_path=npy_path, idx=idx, _logger=logger)
                    predict_details = list()

            render_to_html(webpage=webpage, predict_details=predict_details,
                           npy_path=npy_path, idx=idx, _logger=logger)


def errorchecker(parameters):
    _gpu = parameters.gpu
    _Configure = cfg.ConfigErrorChecker(_gpu)
    # save the configure as the yaml format
    _Configure.save_cfg()
    # Get configures for the project
    _config = _Configure._configs
    # pprint the configure
    pprint(_config)
    logger = init_logger.get_logger(
        _loggerDir=_config.model.log_dir, log_path=_config.model.log_file_name,
        logger_name=_config.model.log_name)

    logger.info('Logging is working ...')
    # Generate the vocab
    _vocab = cfg.VocabErrorChecker(_config, logger)

    if parameters.mode == 'trainval':
        logger.info('This is an trainval mode')
        _TrainDataLoader = DataIteratorErrorChecker(_config, logger, ['train'])
        _ValDataLoader = DataIteratorErrorChecker(_config, logger, ['validate'])
        Model = ErrorCheckerModel(config=_config, vocab=_vocab, logger=logger,
                                  trainable=True)
        Model.build_traineval()
        Model.trainval(_TrainDataLoader, _ValDataLoader)

    elif parameters.mode == 'val':
        """ Validate the dataset """
        logger.info('This is an validate mode')
        # _ValDataLoader = DataIteratorSeq2SeqAtt(_config, logger, ['train', 'validate', 'test'])
        _ValDataLoader = DataIteratorErrorChecker(_config, logger, ['validate'])
        Model = ErrorCheckerModel(config=_config, vocab=_vocab, logger=logger,
                                  trainable=False)
        Model.build_traineval()
        scores = Model.evaluate(_ValDataLoader)
        print('evaluation score is:', scores)

    elif parameters.mode == 'test':
        """ run the test dataset """
        logger.info('This is an test model')
        _TestDataLoader = DataIteratorErrorChecker(_config, logger, ['test'])

        Model = ErrorCheckerModel(config=_config, vocab=_vocab, logger=logger, trainable=False)
        Model.build_inference()
        _, predict_npy, _ = Model.test(_TestDataLoader)

        # np.save('./temp/predict/predict_out', predict_npy)
    else:
        logger.info(
            'This is an predict model, you can predict a sigle image predict files under a given directory')
        temp_path = os.path.abspath(_config.predict.temp_path)
        npy_path = os.path.abspath(_config.predict.npy_path)
        preprocess_dir = os.path.abspath(_config.predict.preprocess_dir)
        render_path = os.path.abspath(_config.predict.render_path)
        webpath = os.path.abspath(_config.predict.web_path)

        _Configure.create_dir(
            [temp_path, npy_path, preprocess_dir, render_path, webpath])

        webpage = cfg.HTML(webpath, title='Predict_Image_Display')
        target_height = _vocab.target_height
        bucket_size = _vocab.bucket_size
        Model = ErrorCheckerModel(config=_config, vocab=_vocab, logger=logger, trainable=False)
        Model.build_inference()
        _ = Model.restore_session()

        while True:
            """ Load the model and restore the weights """

            """ if there is no predict_image given, then input the image_path"""
            if parameters.predict_image:
                image_path = parameters.predict_image
            else:
                # python3
                image_path = input("Input file or directory want to be predict ... >")
                # delete the whitespace
                image_path = simplify(text=image_path)
                if image_path.lower() == 'exit':
                    break
            if os.path.isdir(image_path):
                image_list = get_img_list(image_path)
                logger.info('Total image num want to predict {}'.format(len(image_list)))
            elif os.path.isfile(image_path):
                root, name = os.path.split(image_path)
                image_list = [name]
                image_path = root
            else:
                logger.warn('Please check the input image path')
                continue

            if not image_list:
                logger.warn('Image list is None')
                continue

            predict_details = list()
            image_nums = len(image_list)
            for idx, img_name in enumerate(image_list):
                if img_name.split('.')[-1] == 'pdf':
                    convert_cmd = "magick convert -density {} -quality {} {} {}".format(
                        200, 100, os.path.join(image_path, img_name), os.path.join(
                            preprocess_dir, "{}.png".format(img_name.split('.')[0])))
                    run(cmd=convert_cmd, timeout_sec=TIMEOUT)
                else:
                    src_dir = os.path.join(image_path, img_name)
                    dst_dir = os.path.join(preprocess_dir, img_name)
                    shutil.copy(src=src_dir, dst=dst_dir)
                # preprocess the image
                _img_name_no_ext = img_name.split('.')[0]
                image_process_flage = image_process(
                    input_dir=preprocess_dir, preprocess_dir=preprocess_dir,
                    render_out=render_path, file_name=img_name, target_height=target_height,
                    bucket_size=bucket_size, _logger=logger)

                if not image_process_flage:
                    continue
                image_data = generate_image_data(os.path.join(
                    preprocess_dir, _img_name_no_ext+'.png'), logger, False)
                # predict the image based image data
                _predict_latex_list = Model.predict_single_img(image_data)
                _LatexWant = _predict_latex_list[0]
                # get the directory for the pwd file
                pwd = os.path.abspath(os.getcwd())
                # switch the directory to the render path
                if render_path not in pwd:
                    os.chdir(render_path)
                render_flag = latex_to_image(_LatexWant, _img_name_no_ext, logger)
                # switch directory to the pwd
                os.chdir(pwd)
                if render_flag:
                    param_croped = (
                        os.path.join(render_path, _img_name_no_ext+'.png'),
                        render_path, _img_name_no_ext+'.png', logger)
                    _ = crop_image(param_croped)
                temp = collections.OrderedDict()
                temp['input_dir'] = os.path.join('preprocess', _img_name_no_ext+'.png')
                temp['predict_latex'] = _LatexWant
                temp['render_dir'] = os.path.join(
                    'render', _img_name_no_ext + '.png') if os.path.exists(
                    os.path.join(render_path, _img_name_no_ext + '.png')) else None
                predict_details.append(temp)

                if idx % 200 == 0 and idx != 0:
                    render_to_html(webpage=webpage, predict_details=predict_details,
                                   npy_path=npy_path, idx=idx, _logger=logger)
                    predict_details = list()

            render_to_html(webpage=webpage, predict_details=predict_details,
                           npy_path=npy_path, idx=idx, _logger=logger)


if __name__ == "__main__":
    while True:
        im2katex()
        errorchecker()
