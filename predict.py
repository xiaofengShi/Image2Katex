from scipy.misc import imread


from model.utils.general import Config, run
from model.utils.text import Vocab
from model.utils.image import greyscale, crop_image, pad_image, \
    downsample_image, TIMEOUT

import argparse
import collections
import csv
import json
import os
import pickle
import sys
from pprint import pprint

import numpy as np
import pandas as pd
from PIL import Image

import config
import init_logger
from data.data_utils import crop_image, img_padding
from dataset_iter import DataIterator
from models.seq2seq_model import Seq2SeqAttModel
from models.utils.general import init_dir, run
from models.utils.image import (IMG_EXTENSIONS, TIMEOUT, get_img_list,
                                resize_img_target_height, render_png)
from models.utils.render_image import latex_to_image









def interactive_shell(model):
    """Creates interactive shell to play with model
    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
Enter a path to a file
input> data/images_test/0.png""")

    while True:
        try:
            # for python 2
            img_path = raw_input("input> ")
        except NameError:
            # for python 3
            img_path = input("input> ")

        if img_path == "exit":
            break

        if img_path[-3:] == "png":
            img = imread(img_path)

        elif img_path[-3:] == "pdf":
            # call magick to convert the pdf into a png file
            buckets = [
                [240, 100], [320, 80], [400, 80], [400, 100], [480, 80], [480, 100],
                [560, 80], [560, 100], [640, 80], [640, 100], [720, 80], [720, 100],
                [720, 120], [720, 200], [800, 100], [800, 320], [1000, 200],
                [1000, 400], [1200, 200], [1600, 200], [1600, 1600]
            ]

            dir_output = "tmp/"
            name = img_path.split('/')[-1].split('.')[0]
            run("magick convert -density {} -quality {} {} {}".format(200,
                                                                      100, img_path, dir_output+"{}.png".format(name)), TIMEOUT)
            img_path = dir_output + "{}.png".format(name)
            crop_image(img_path, img_path)
            pad_image(img_path, img_path, buckets=buckets)
            downsample_image(img_path, img_path, 2)

            img = imread(img_path)

        img = greyscale(img)
        hyps = model.predict(img)

        model.logger.info(hyps[0])


if __name__ == "__main__":
    # restore config and model

    parameters = process_args(args)

    _config = config.Config().config
    pprint(_config)

    logger = init_logger.get_logger(
        _loggerDir=_config.model.model_saved, log_path=parameters.log_path,
        logger_name=parameters.logger_name)
    logger.info('Logging is working ...')

    _vocab = config.Vocab(_config, logger)

    logger.info('Load trainer data done ...')
    
    Model = Seq2SeqAttModel(config=_config, vocab=_vocab, logger=logger, trainable=False)
    Model.build_inference()
    _ = Model.restore_session()

    interactive_shell(Model)
