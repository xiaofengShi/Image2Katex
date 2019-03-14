# -*- coding: utf-8 -*-
import argparse
import copy
import json
import os
import re
import sys
import config as cfg
import init_logger

import time
from load_model import LoadModel
from collections import OrderedDict
from PIL import Image
import glob
# from config import ConfigApi as cfg
from flask import Flask, render_template, request, send_from_directory, url_for
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from werkzeug import secure_filename
from werkzeug import SharedDataMiddleware

# import load_model

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'pdf'])

app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = './temp/imgs/uploaded'
app.config['RENDER_FOLDER'] = './static/render'
app.config['PROCESS_FOLDER'] = './static/preprocess'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# app.add_url_rule('/uploads/<filename>', 'uploaded_file',
#                  build_only=True)
# app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
#     '/uploads':  app.config['UPLOAD_FOLDER']})

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


@app.route('/predict', methods=['GET', 'POST'])
def see_predict():
    return render_template('predict.html')


@app.route('/predict200', methods=['GET', 'POST'])
def see_predict200():
    return render_template('predict_200.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/rendered_img/<filename>', methods=['GET', 'POST'])
def rendered_img(filename):
    return send_from_directory(app.config['RENDER_FOLDER'], filename)


@app.route('/process_img/<filename>', methods=['GET', 'POST'])
def process_img(filename):
    return send_from_directory(app.config['PROCESS_FOLDER'], filename)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = str(time.time()).replace('.', '')+'-'+secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('uploaded_file', filename=filename)
            width = min([Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename)).size[0], 600])
            predict_details = Moedl.run_im2latex(
                os.path.join(app.config['UPLOAD_FOLDER'], filename))
            latex = predict_details['predict_latex']

            out_img_name = predict_details['render_dir']

            _process_img = url_for('process_img', filename=predict_details['process_img'])

            if out_img_name is not None:
                render_image = url_for('rendered_img', filename=out_img_name)
            else:
                render_image = []

            return render_template(
                'index.html', img=file_url, process_img=_process_img, width=width, text=latex,
                render_imgs=render_image)

    return render_template('index.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Define the mode for the process'))
    parser.add_argument(
        '--data_type', dest='data_type', default='merged',
        choices={'handwritten', 'original', 'merged'},
        help=('The dataset want to be trained'))
    parser.add_argument(
        '--gpu', dest='gpu', default=0,
        choices={0, -1}, type=int,
        help=('Whether use gpu or not'))

    parameters = parser.parse_args()

    _dataset_type = parameters.data_type
    _gpu = parameters.gpu
    _Configure = cfg.ConfigSeq2Seq(_dataset_type, _gpu)
    # save the configure as the yaml format
    # Get configures for the project
    _config = _Configure._configs

    # pprint the configure
    # Generate the logger
    logger = init_logger.get_logger(
        _loggerDir='./static', log_path='server.log',
        logger_name='server')
    logger.info('Server is working ...')
    # Generate the vocab
    _vocab = cfg.VocabSeq2Seq(_config, logger)

    Moedl = LoadModel(ConfClass=_Configure, _config=_config,
                      _vocab=_vocab, logger=logger, trainable=False)

    print('Load models done ... ...')

    app.run(host='0.0.0.0', port=9999, debug=True)
