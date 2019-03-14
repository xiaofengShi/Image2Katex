

'''
File: generate_sequence.py
Project: utils
File Created: Monday, 24th December 2018 12:23:26 pm
Author: xiaofeng (sxf1052566766@163.com)
-----
Last Modified: Monday, 24th December 2018 12:23:37 pm
Modified By: xiaofeng (sxf1052566766@163.com>)
-----
Copyright 2018.06 - 2018 onion Math, onion Math
'''
from __future__ import absolute_import, division, print_function

import collections
import json
import os
import pickle
import re
import sys

import random
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow.python.platform import gfile
from collections import defaultdict
from config_dataset import SequenceVocabulary
from get_logger import init_logger

sys.path.append('..')
sys.path.insert(0, os.path.dirname(os.path.abspath(os.getcwd())))
print(sys.path)
from models.evaluate.text import cal_score
from utils.render_image import latex_to_image

""" This progress is designed to generate the sequence that can not generate the png file  
- 使用公式预测网络预测生成latex储存到对应的文件夹中，同时，存在真实的label，本程序主要为了进行nmt网络的搭建
，使用预测的latex作为源输入，使用对应的label作为目标输出nmt
"""


class ErrorChecker(object):
    def __init__(self, logger, vocabulary):

        self.logger = logger
        self.vocabulary = vocabulary
        self.token_to_idx = self.vocabulary.token_to_idx
        self.idx_to_token = self.vocabulary.idx_to_token
        self.bucket = self.vocabulary.bucket_size
        self.logger.info('Process the ErrorChecker function')

    def readfile(self, files):
        if files.endswith('.txt'):
            return [i.strip().split() for i in open(files, 'r').readlines()]
        elif files.endswith('.dat') or files.endswith('.pkl'):
            return pickle.load(open(files, 'rb'))

    def readfilenums(self, files):
        if files.endswith('.txt'):
            return [len(i.strip().split()) for i in open(files, 'r').readlines()]
        elif files.endswith('.dat') or files.endswith('.pkl'):
            file_list = pickle.load(open(files, 'rb'))
            return [len(i) for i in file_list]

    def writefile(self, input_data, files):

        if files.endswith('.txt'):
            with open(files, 'w') as wr:
                for line in input_data:
                    wr.write(' '.join([str(j) for j in line]) + '\n')
        elif files.endswith('.dat') or files.endswith('.pkl'):
            with open(files, 'wb') as wr:
                pickle.dump(input_data, wr, True)

    def fit_plot_kmeans_model(self, n, X):
        """ Kmeans to predict the bucket size """
        # print('clustering kmeans ...')
        kmean = KMeans(n_clusters=n, max_iter=1000, tol=0.01, init='k-means++', n_jobs=-1)
        kmean.fit(X)
        # print('kmenas: k={} ,cost={}'.format(n, int(kmean.inertia_)))
        # print('centers: {}'.format(kmean.cluster_centers_))
        return kmean.cluster_centers_, kmean.inertia_

    def _cal_buckets(
            self, source_file, target_file, bucket_file, prepared_dir, min_val=4, max_val=20):
        # cal buckets based the kmeans
        self.logger.info('Cal the bucket size ...')
        with open(bucket_file, 'w')as k:
            x = self.readfilenums(source_file)
            y = self.readfilenums(target_file)
            self.logger.info('source line nums is [{}]'.format(len(x)))
            self.logger.info('target line nums is [{}]'.format(len(y)))
            data = np.stack([x, y], axis=1)
            temp = dict()
            loss = []
            for i in range(min_val, max_val):
                centers, distance = self.fit_plot_kmeans_model(i, data)
                centers = centers.tolist()
                centers_ori = [[int(i[0]), int(i[1])] for i in centers]
                centers_sort = sorted(centers_ori, key=lambda k: k[0])
                temp[str(i)+'_source_target'] = centers_sort
                temp[str(i) + '_distance'] = distance / len(x)
                loss.append(distance/len(x))
            self.logger.info('Save bucket details to the file [{}]'.format(bucket_file))
            json.dump(temp, k)
        self.plot_scatter_lengths(
            title='loss', x_title='k_iter', y_title='distance',
            x_lengths=list(range(min_val, max_val)),
            y_lengths=loss, out_file=prepared_dir)

    def _cal_score(self, source_file, target_file):
        """ calculate score between predict and target """
        score = cal_score(source_file, target_file)
        out = {}
        out['description'] = u"The evaluation score for the predict and label"
        out['evaluation'] = score
        with open('score.json', 'w') as js:
            json.dump(out, js)

    def plot_scatter_lengths(self, title, x_title, y_title, x_lengths, y_lengths, out_file):
        plt.figure()
        plt.scatter(x_lengths, y_lengths)
        plt.title(title)
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.ylim(0, max(y_lengths))
        plt.xlim(0, max(x_lengths))
        # plt.show()
        plt.savefig(os.path.join(out_file, '{}.png'.format(title)))

    def plot_histo_lengths(self, title, lengths):
        plt.figure()
        mu = np.std(lengths)
        sigma = np.mean(lengths)
        x = np.array(lengths)
        n, bins, patches = plt.hist(x,  50, facecolor='green', alpha=0.5)
        y = scipy.stats.norm.pdf(bins, mu, sigma)
        plt.plot(bins, y, 'r--')
        plt.title(title)
        plt.xlabel("Length")
        plt.ylabel("Number of Sequences")
        plt.xlim(0, max(lengths))
        plt.savefig('{}.png'.format(title))
        # plt.show()

    def analysisfile(self, source_file, target_file, figure_file, plot_histograms=True,
                     plot_scatter=True):
        """ Anaylsis and display the file """
        source_lengths = []
        target_lengths = []

        with gfile.GFile(source_file, mode="r") as s_file:
            with gfile.GFile(target_file, mode="r") as t_file:
                source = s_file.readline()
                target = t_file.readline()
                counter = 0

                while source and target:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  reading data line %d" % counter)
                        sys.stdout.flush()
                    num_source_ids = len(source.split())
                    source_lengths.append(num_source_ids)
                    num_target_ids = len(target.split()) + 1  # plus 1 for EOS token
                    target_lengths.append(num_target_ids)
                    source, target = s_file.readline(), t_file.readline()
        # print(target_lengths, source_lengths)
        if plot_histograms:
            self.plot_histo_lengths("target lengths", target_lengths)
            self.plot_histo_lengths("source_lengths", source_lengths)
        if plot_scatter:
            self.plot_scatter_lengths("target vs source length", "source length",
                                      "target length", source_lengths, target_lengths, figure_file)

    def merge_sequence(self, source_dir, target_dir, write_source, write_target):
        """ merge the files that predicted latex and label latex into one file  """
        self.logger.info('Merge the files ...')
        # 将验证的多个label和predict进行合并成一个单独文件，使用一个机器翻译的模型进行错误检测及纠正
        assert os.path.exists(source_dir), '[{}] do not exist'.format(source_dir)
        source_file_list = self.getRawFileList(source_dir)
        target_file_list = self.getRawFileList(target_dir)
        merged_source, merged_target = [], []
        for idx in range(len(source_file_list)):
            _source_file = source_file_list[idx]
            _child = _source_file.split('/')[-2]
            _target_file = [i for i in target_file_list if i.split('/')[-2] == _child][0]
            print(_source_file)
            print(_target_file)
            _source_details = self.readfile(_source_file)
            _target_details = self.readfile(_target_file)
            assert len(_source_details) == len(_target_details), ' sequence num must be same'
            merged_source.extend(_source_details)
            merged_target.extend(_target_details)
        self.logger.info('Source file nums is [{:d}]'.format(len(merged_source)))
        self.logger.info('Target file nums is [{:d}]'.format(len(merged_target)))
        self.writefile(merged_source, write_source)
        self.writefile(merged_target, write_target)
        self.logger.info('Merge the files done')

    def convert_char_idx(self, input_file, out_file):
        """ Convert the char to the idx based the vocabulayer dictionary """
        out = []
        missing = {}
        self.logger.info('Convert the char file to ids file for the [{}]'.format(input_file))
        with tf.gfile.GFile(input_file, mode='r') as ip:
            source = ip.readline().strip()
            counter = 0
            while source:
                counter += 1
                if counter % 1000 == 0:
                    print('Reanding data line %d' % counter)
                source_list = source.split()
                temp = [self.vocabulary.START_ID]
                for char in source_list:
                    try:
                        temp += [self.token_to_idx[char]]
                    except:
                        if char not in missing.keys():
                            missing[char] = 0
                        missing[char] += 1
                        temp += [self.vocabulary.UNK_ID]
                temp += [self.vocabulary.EOS_ID]
                out.append(temp)
                source = ip.readline()
        self.logger.info(' missing char is {}:'.format(missing.keys()))
        with open(out_file, 'w') as ou:
            for i in out:
                ou.write(' '.join([str(j) for j in i]) + '\n')

    def getRawFileList(self, path):
        files = []
        for f in os.listdir(path):
            if os.path.isfile(os.path.join(path, f)) and f.endswith('.txt'):
                files.append(os.path.join(path, f))
            if os.path.isdir(os.path.join(path, f)):
                temp = self.getRawFileList(os.path.join(path, f))
                files.extend(temp)
        return files

    def generata_sequence_dataset(
            self, source_path, target_path, dataset_file):
        # sorte the size based the target size
        data_set = defaultdict(list)
        with tf.gfile.GFile(source_path, mode="r") as source_file:
            with tf.gfile.GFile(target_path, mode="r") as target_file:
                source, target = source_file.readline().strip(), target_file.readline().strip()
                counter = 0
                while source and target:
                    counter += 1
                    if counter % 100000 == 0:
                        print(" reading data line %d" % counter)
                        sys.stdout.flush()
                    source_ids = [int(x) for x in source.split()]
                    target_ids = [int(x) for x in target.split()]
                    for bucket_id, (source_size, target_size) in enumerate(self.bucket):
                        # if str(self.bucket[bucket_id]) not in data_set:
                        #     data_set[str(self.bucket[bucket_id])] = []
                        if len(source_ids) < source_size and len(target_ids) < target_size:
                            # random droupout for the souce sequence
                            source_length = len(source_ids)
                            _source_idx = source_ids
                            if random.random() > 0.5:
                                droupout = random.randrange(source_length)
                                _source_idx = source_ids[0:droupout] + \
                                    source_ids[droupout + 1:source_length]
                            data_set[self.bucket[bucket_id]].append([_source_idx, target_ids])
                            break
                    source, target = source_file.readline(), target_file.readline()
        np.save(os.path.join(dataset_file, 'ErrorChecker_dataset'), data_set)

        self.logger.info('Saving dataset to [{}]'.format(
            os.path.join(dataset_file, 'ErrorChecker_dataset')))
        del data_set

    def split_train_val_test(self, numpy_datapath, dataset_file):
        self.logger.info('Split train test and validate')
        dataset_details = np.load(numpy_datapath).tolist()
        key_list = dataset_details.keys()
        train_perp = 0.9
        test_perp = 0.98
        train_dataset, test_dataset, val_dataset = defaultdict(
            list), defaultdict(list), defaultdict(list)
        for key in key_list:
            bucket_details = dataset_details[key]
            nums = len(bucket_details)
            # shufull
            random.shuffle(bucket_details)
            _train_num = int(train_perp * nums)
            _test_num = int(test_perp*nums)
            train_dataset[key].extend(bucket_details[:_train_num])
            test_dataset[key].extend(bucket_details[_train_num:_test_num])
            val_dataset[key].extend(bucket_details[_test_num:])
        np.save(os.path.join(dataset_file, 'train_buckets'), train_dataset)
        np.save(os.path.join(dataset_file, 'test_buckets'), test_dataset)
        np.save(os.path.join(dataset_file, 'validate_buckets'), val_dataset)
        self.logger.info('Split train, test and validate done...')
        del train_dataset, test_dataset, val_dataset

    def rendered_filter(
            self, source_token, target_token, render_path, source_filtered_path,
            target_filtered_path):
        filtered_data = defaultdict(list)
        source_filtered_token, target_filtered_token = [], []
        # current path
        pwd = os.path.abspath(os.getcwd())
        # switch the directory to the render path
        render_path = os.path.abspath(render_path)
        source_token_list = open(source_token).readlines()
        target_token_list = open(target_token).readlines()
        assert len(source_token_list) == len(
            target_token_list), 'The length of source and target must be same'
        nums = len(source_token_list)
        for idx in range(nums):
            source = source_token_list[idx].strip()
            target = target_token_list[idx].strip()
            if render_path not in pwd:
                os.chdir(render_path)
            render_flag = latex_to_image(source, str(idx), self.logger)
            # switch to th current path
            os.chdir(pwd)
            # 如果可渲染成功，跳过
            if render_flag:
                render_img = os.path.join(render_path, str(idx) + '.png')
                assert os.path.exists(
                    render_img), 'do not exist the file [{:s}]'.format(render_img)
                os.remove(render_img)
                continue
            else:
                source_filtered_token.append(source)
                target_filtered_token.append(target)
        assert len(source_filtered_token) == len(
            target_filtered_token), 'Filter token nums must be same'
        self.writefile(source_filtered_token, source_filtered_path)
        self.writefile(target_filtered_token, target_filtered_path)


if __name__ == "__main__":
    logger = init_logger(log_path='sequence_dataset.log', logger_name='ErrorCheck')
    logger.info('Load logger done...')
    preprocess = ErrorChecker(logger=logger, vocabulary=SequenceVocabulary)
    prepared_dir = './errorchecker_dataset/prepared'
    temp_dir = './errorchecker_dataset/temp'
    source_dir = './errorchecker_dataset/eval_files_from_im2latex/predict'
    traget_dir = './errorchecker_dataset/eval_files_from_im2latex/label'
    write_source_token = './errorchecker_dataset/temp/merged_source_token.txt'
    # 对源序列进行过滤，只剩下不能渲染图片的序列
    source_filtered_path = './errorchecker_dataset/temp/filtered_source_token.txt'
    # 对目标序列进行过滤，保证和过滤之后的源序列数量相同
    target_filtered_path = './errorchecker_dataset/temp/filtered_target_token.txt'
    write_source_ids = './errorchecker_dataset/temp/merged_source_ids.txt'
    write_target_token = './errorchecker_dataset/temp/merged_target_token.txt'
    write_target_ids = './errorchecker_dataset/temp/merged_target_ids.txt'
    buckets_file = './errorchecker_dataset/prepared/buckets.json'
    train_dataset = './errorchecker_dataset/prepared/train_buckets.npy'
    render_path = './errorchecker_dataset/rendered'

    # merged the sequence for the datasset
    if not os.path.exists(write_source_token) and not os.path.exists(write_target_token):
        preprocess.merge_sequence(source_dir, traget_dir, write_source_token, write_target_token)
    # convert the char to the idx

    # filter the sequence
    if not os.path.exists(source_filtered_path) or not os.path.exists(target_filtered_path):

        preprocess.rendered_filter(write_source_token, write_target_token,
                                   render_path, source_filtered_path, target_filtered_path)

    """ # source file
    if not os.path.exists(write_source_ids):
        preprocess.convert_char_idx(input_file=write_source_token, out_file=write_source_ids)
    # target file
    if not os.path.exists(write_target_ids):
        preprocess.convert_char_idx(input_file=write_target_token, out_file=write_target_ids)
    # cal buckets
    if not os.path.exists(buckets_file):
        preprocess._cal_buckets(write_source_token, write_target_token, buckets_file, prepared_dir)
    # Generate dataset for the numpy format
    if not os.path.exists(os.path.join(temp_dir, 'ErrorChecker_dataset.npy')):
        preprocess.generata_sequence_dataset(
            source_path=write_source_ids, target_path=write_target_ids, dataset_file=temp_dir)
    # Split train test and validate
    if not os.path.exists(train_dataset):
        preprocess.split_train_val_test(os.path.join(
            temp_dir, 'ErrorChecker_dataset.npy'), prepared_dir) """
