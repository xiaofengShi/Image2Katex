'''
Filename: trainer.py
Project: image2katex
File Created: Wednesday, 5th December 2018 5:20:12 pm
Author: xiaofeng (sxf1052566766@163.com)
--------------------------
Last Modified: Wednesday, 5th December 2018 5:20:32 pm
Modified By: xiaofeng (sxf1052566766@163.com)
---------------------------
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
from RunModel import im2katex,errorchecker


def process_args(args):
    parser = argparse.ArgumentParser(description=('Define the mode for the process'))

    parser.add_argument(
        '--model_type', dest='model_type', default='im2katex',
        choices={'im2katex', 'error'}, required=True,
        help=('Which model want to construct and run'))
    
    parser.add_argument(
        '--gpu', dest='gpu', default=0,
        choices={0,-1}, type=int,
        help=('Whether use gpu or not'))

    parser.add_argument('--mode', dest='mode',
                        default='train', choices={'trainval', 'test', 'val', 'infer'},
                        required=True, help=('The mode for the process'))

    parser.add_argument('--encoder_type', dest='encoder_type',
                        default='conv', choices={'Augment','conv'},
                         help=('The encoedr type of the model'))

    parser.add_argument(
        '--data_type', dest='data_type', default='merged',
        choices={'handwritten', 'original', 'merged'},
        help=('The dataset want to be trained for the im2katex model'))

    parser.add_argument(
        '--predict_img_path', dest="predict_image", type=str,
        help=(
            'The image path directory want to predict for the inference model of the im2katex model'))

    parser.add_argument('--log_path', dest="log_path",
                        type=str, default='runing.log',
                        help=('Log file path, default=runing.log'))
    parser.add_argument('--logger_name', dest="logger_name",
                        type=str,
                        default='image2latex', choices={'image2latex', 'ErrorChecker'},
                        help=('logger name, choose from [image2latex ,ErrorChecker]'))

    parameters = parser.parse_args(args)
    return parameters


def main(args):
    parameters = process_args(args)
    if parameters.model_type == 'im2katex':
        im2katex(parameters)
    else:
        errorchecker(parameters)


if __name__ == "__main__":

    main(sys.argv[1:])
