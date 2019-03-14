

'''
Filename: data_utils.py
Project: dataset
File Created: Friday, 30th November 2018 5:30:42 pm
Author: xiaofeng (sxf1052566766@163.com)
--------------------------
Last Modified: Friday, 30th November 2018 5:32:08 pm
Modified By: xiaofeng (sxf1052566766@163.com)
---------------------------
Copyright: 2018.06 - 2018 OnionMath. OnionMath
'''

import copy
import csv
import datetime
import json
import os
import pickle
import platform
import random
import re
import shutil

import numpy as np
import pandas as p
from keras.preprocessing.image import (ImageDataGenerator, array_to_img,
                                       img_to_array, load_img)
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from tqdm import tqdm

import cv2

# import matplotlib
# system_version = platform.system()
# if system_version == 'Darwin':
#     matplotlib.use('TkAgg')
# if system_version == 'Linux':
#     matplotlib.use('Agg')
# from matplotlib import pyplot as plt


""" image processing programs """


def extract_peek_ranges_from_array(
        array_vals, minimun_val=100, minimun_range=30):
    """ 对单个字符进行分割 """
    start_i = None
    end_i = None
    peek_ranges = []
    for i, val in enumerate(array_vals):
        if val > minimun_val and start_i is None:
            start_i = i
        elif val > minimun_val and start_i is not None and i != len(array_vals)-1:
            pass
        elif val < minimun_val and start_i is not None:
            end_i = i
            if end_i - start_i >= minimun_range:
                peek_ranges.append((start_i, end_i))
            start_i = None
            end_i = None
        elif val > minimun_val and i == len(array_vals)-1 and start_i is not None:
            end_i = len(array_vals)-1
            peek_ranges.append((start_i, end_i))

        elif val < minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")
    return peek_ranges


def crop_image(l):
    """ use Image to crop the image """
    img, crop_saved_path, file_name = l
    try:

        old_im = Image.open(img).convert('L')
        witdth, height = old_im.size
        img_data = np.asarray(old_im, dtype=np.uint8)  # height, width
        nnz_inds = np.where(img_data != 255)
        if len(nnz_inds[0]) == 0:
            return None
        # """ 只保存含有一行文本内容的图片，对于多行文字内容跳过 """
        # _, adaptive_threshold = cv2.threshold(img_data, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # horizontal_sum = np.sum(adaptive_threshold, axis=1)

        # """ 对竖直方向的每行进行分割，对于题目是左右结构并且不对齐的情况存在问题 """
        # peek_ranges = extract_peek_ranges_from_array(horizontal_sum, minimun_val=10, minimun_range=5)
        # if len(peek_ranges) != 1:
        #     return None
        y_min = np.min(nnz_inds[0])
        y_max = np.max(nnz_inds[0])
        x_min = np.min(nnz_inds[1])
        x_max = np.max(nnz_inds[1])
        if (x_max - x_min) * (y_max - y_min) < 100:
            return None
        old_im = old_im.crop((x_min, y_min, x_max + 1, y_max + 1))
        # NOTE: 查看截取的图像是否具有多行
        croped_height = y_max - y_min
        old_img_numpy = np.asarray(old_im, dtype=np.uint8)
        _, adaptive_threshold = cv2.threshold(
            old_img_numpy, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        horizontal_sum_croped = np.sum(adaptive_threshold, axis=1)
        peek_ranges_croped = extract_peek_ranges_from_array(
            horizontal_sum_croped, minimun_val=10, minimun_range=5)
        if len(peek_ranges_croped) != 1:
            return None
        if int((peek_ranges_croped[0][-1]) * 1.5) < croped_height:
            return None
        old_im.save(os.path.join(crop_saved_path, file_name))
        out_name = [file_name]
        return out_name
    except Exception as e:
        print('error:', e)
        return None


def crop_image_filter(l):
    """ use Image to crop the image """
    img, crop_saved_path, file_name = l
    try:
        old_im = Image.open(img).convert('L')
        witdth, height = old_im.size
        img_data = np.asarray(old_im, dtype=np.uint8)  # height, width
        nnz_inds = np.where(img_data != 255)
        if len(nnz_inds[0]) == 0:
            return None
        y_min = np.min(nnz_inds[0])
        y_max = np.max(nnz_inds[0])
        x_min = np.min(nnz_inds[1])
        x_max = np.max(nnz_inds[1])
        if (x_max - x_min) * (y_max - y_min) < 100:
            return None
        # NOTE: 查看截取的图像是否具有多行
        old_im = old_im.crop((x_min, y_min, x_max + 1, y_max + 1))
        old_img_numpy = np.asarray(old_im, dtype=np.uint8)
        croped_height = y_max - y_min
        _, adaptive_threshold = cv2.threshold(
            old_img_numpy, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        horizontal_sum_croped = np.sum(adaptive_threshold, axis=1)
        peek_ranges_croped = extract_peek_ranges_from_array(
            horizontal_sum_croped, minimun_val=10, minimun_range=5)
        if len(peek_ranges_croped) != 1:
            # print('peek_ranges_croped', peek_ranges_croped)
            return None
        if int((peek_ranges_croped[0][-1]) * 1.5) < croped_height:
            print('not ...')
            return None
        old_im.save(os.path.join(crop_saved_path, file_name))
        out_name = [file_name]
        return out_name
    except Exception as e:
        print('error:', e)
        return None


# from multiprocessing import Pool
# img_ori = '/home/xiaofeng/data/formula/origin/img_filtered'
# croped_dir = '/home/xiaofeng/data/formula/origin/img_crop'
# croped_img_names = './formula/dataset/temp/croped_imgname.pkl'
# if not os.path.exists(croped_dir):
#     print('Crop the ori img...')
#     os.makedirs(croped_dir)
# filelist = [i for i in os.listdir(img_ori) if i.endswith('.png')]
# print('all img nums:', len(filelist), 'files saved to:{}'.format(croped_dir))
# pool = Pool(14)
# out_names = list(
#     pool.map(
#         crop_image_filter,
#         [(os.path.join(img_ori, filename),
#           croped_dir, filename) for filename in filelist]))
# with open(croped_img_names, 'wb') as ou:
#     pickle.dump(out_names, ou, pickle.HIGHEST_PROTOCOL)
# pool.close()
# pool.join()

def size_bucket_modify(size_bucket_kmeans, height_target):
    height_list = [i[-1] for i in size_bucket_kmeans]
    width_list = [i[0] for i in size_bucket_kmeans]
    out = []
    for i in range(len(height_list)):
        cur_height = height_list[i]
        cur_width = width_list[i]
        distance = [abs(x - cur_height) for x in height_target]
        find = [i for i, j in enumerate(distance) if j == min(distance)][-1]
        target_height = height_target[find]
        sacle = cur_height / target_height
        target_width = int(cur_width / sacle)
        new_size = (target_width, target_height)
        out.append(new_size)
    return out


def get_img_size(input_dir, size_ori_saved):
    """ get the size of the image ang saved them to pkl """
    img_list = [i for i in os.listdir(input_dir) if i.endswith('.png')]
    size = set()
    for img in img_list:
        img_path = os.path.join(input_dir, img)
        img_shp = Image.open(img_path).size
        size.add(img_shp)
    size = list(size)
    with open(size_ori_saved, 'wb') as si:
        pickle.dump(size, si, pickle.HIGHEST_PROTOCOL)
    return size


def k_means_size_list(
        input_dir, size_ori_saved, size_bucket, height_list, loop_nums=2000, display=False):
    """ k-means to find the proper size list """
    """ load the ori size list of the img """
    if not os.path.exists(size_ori_saved):
        # print('Get the size of the croped images from {}'.format(input_dir))
        size_list = get_img_size(input_dir, size_ori_saved)
    else:
        with open(size_ori_saved, 'rb') as sz:
            size_list = pickle.load(sz)

    size_bucket_list = []
    for target_height in height_list:
        width = [size[0] for size in size_list if size[1] == target_height]
        height = [size[1] for size in size_list if size[1] == target_height]

        max_width_size = [max(width), target_height]

        p_list = np.stack([width, height], axis=1)
        """ calculate the proper K """
        loss = []
        K = 15
        if len(p_list) < K:
            continue
        for i in range(1, K):
            kmeans = KMeans(n_clusters=i, max_iter=1000).fit(p_list)
            # 每个聚类中样本到其聚类中心的平方距离之和
            loss.append(kmeans.inertia_ / len(width))

        loss_diff = [loss[idx] - loss[idx + 1] for idx in range(len(loss) - 1)]
        for idx in range(len(loss_diff)):
            if loss_diff[idx] < 1:
                break
        # k_idx = loss_diff.index(max(loss_diff))
        k_idx = idx+1
        if display:
            plt.figure('loss')
            plt.subplot(121)
            plt.scatter(range(1, 30), loss, marker='*', c='r')

            plt.subplot(122)
            plt.scatter(range(1, 29), loss_diff, marker='*', c='b')
            plt.savefig('./loss.jpg')

        """ calculate the k sizes """
        index = np.random.choice(len(p_list), size=k_idx)
        centeroid = p_list[index]
        for i in range(loop_nums):
            points_set = {key: [] for key in range(k_idx)}
            # print('count/total:{}/{}'.format(i, loop_nums))
            for p in p_list:
                nearest_index = np.argmin(
                    np.sum((centeroid - p) ** 2, axis=1) ** 0.5)
                points_set[nearest_index].append(p)

            for k_index, p_set in points_set.items():
                if len(p_set) == 0:
                    break
                p_xs = [p[0] for p in p_set]
                p_ys = [p[1] for p in p_set]
                centeroid[k_index, 0] = sum(p_xs) / len(p_set)
                centeroid[k_index, 1] = sum(p_ys) / len(p_set)

        width_center = [cen[0] for cen in centeroid]
        height_center = [cen[1] for cen in centeroid]
        centeroid = centeroid.tolist()
        centeroid.append(max_width_size)
        size_bucket_list.extend(centeroid)

        if display:
            plt.figure('display')
            plt.axis()
            plt.scatter(width, height)
            plt.scatter(width_center, height_center, c='r', marker='*')
            plt.savefig('./display.jpg')

    size_bucket_list = size_bucket_modify(size_bucket_list, height_list)
    with open(size_bucket, 'wb') as sz:
        pickle.dump(size_bucket_list, sz, pickle.HIGHEST_PROTOCOL)
    # print('size_bucket_list', size_bucket_list)
    return size_bucket_list


def img_padding(parameters):
    filename, input_dir, size_bucket, padding_dir = parameters
    img_path = os.path.join(input_dir, filename)
    old_im = Image.open(img_path)
    width, height = old_im.size

    """ 只变大不变小 """
    """ buckets 包含resize 之后最大的width尺寸 """
    width_list = [
        [size_bucket[i][0] - width, size_bucket[i]]
        for i in range(len(size_bucket))
        if  size_bucket[i][1] == height and abs(size_bucket[i][0] - width) <= 50 ]
    if width_list:
        temp = [width_select[0] for width_select in width_list]
        target_idx = temp.index(min(temp))
        target_size = width_list[target_idx][1]
        new_im = Image.new("RGB", target_size, (255, 255, 255))
        new_im.paste(old_im)
        new_im.save(os.path.join(padding_dir, filename))


def get_img_size_dict(input_dir, size_ori_dict):
    img_list = [i for i in os.listdir(input_dir) if i.endswith('.png')]
    print('img num:', len(img_list))
    size = set()
    size_dict = {}
    size_dict['size'] = {}
    size_dict['height'] = {}
    size_dict['img_num'] = len(img_list)
    for img in img_list:
        img_path = os.path.join(input_dir, img)
        img_shp = Image.open(img_path).size
        size.add(img_shp)
        if img_shp not in size_dict['size']:
            size_dict['size'][str(img_shp)] = 0
        size_dict['size'][str(img_shp)] += 1

    size_list = list(size)
    print('size_list:', len(size_list))
    for size in size_list:
        if size[-1] not in size_dict['height']:
            size_dict['height'][size[-1]] = 0

        size_dict['height'][size[-1]] += 1
    with open(size_ori_dict, 'w') as siz:
        json.dump(size_dict, siz, ensure_ascii=False)


def resize_img_target_height(l):
    """ resize the img with the height of [10,20,30,40] """
    filename, input_dir, height_dict, resized_img_path, height_list = l

    img_path = os.path.join(input_dir, filename)
    img = Image.open(img_path)
    width, height = img.size
    # print('height:', height)
    assert str(height) in height_dict, 'height must in the height dict'
    if height_dict[str(height)] < 10:
        pass
    distance_list = [abs(height - i) for i in height_list]
    target_height = height_list[distance_list.index(min(distance_list))]
    sacle = height / target_height
    target_width = int(width / sacle)
    new_size = (target_width, target_height)
    resize_img = img.resize(new_size)
    resize_img.save(os.path.join(resized_img_path, filename))


def resize_img_based_buckedt(l):
    """ resize the img based on the size list calculated by k-means """
    filename, size_list, input_dir, resized_img_path = l
    img_path = os.path.join(input_dir, filename)
    img = Image.open(img_path)
    old_size = np.asarray(img.size)
    min_idx = np.argmin(np.sum((size_list - old_size) ** 2, axis=1) ** 0.5)
    target_size = size_list[min_idx]
    resized_img = img.resize(target_size)
    resized_img.save(os.path.join(resized_img_path, filename))


def get_vocabluary(parameters):
    """ generate the vocab dictionary and split the full dataset into train and validate"""
    label_path, voacbulary_path, vocab_dict_path, temp, logger = parameters
    # step1: split the dataset into train and validate based the label_list
    lable_list = [
        i for i in os.listdir(label_path)
        if i.endswith('.ls') or i.endswith('.lst')]
    # step2: make the vocabulary
    vocab_dictionary = {}
    for label_name in lable_list:
        label_file = os.path.join(label_path, label_name)
        label_contails = open(label_file).readlines()
        for label in label_contails:
            tokens = label.strip().split()
            for token in tokens:
                if token not in vocab_dictionary:
                    vocab_dictionary[token] = 0
                vocab_dictionary[token] += 1
    vocab = sorted(list(vocab_dictionary.keys()))

    vocab_out = []
    num_unknown = []
    unk_threshold = 5
    for word in vocab:
        if vocab_dictionary[word] > unk_threshold:
            vocab_out.append(word)
        else:
            num_unknown.append(word)
    logger.info('Write vocabulary text to {:s}'.format(voacbulary_path))
    logger.info('unknown word is {}'.format(num_unknown))
    with open(voacbulary_path, 'w') as fout:
        fout.write('\n'.join(vocab_out))

    with open(vocab_dict_path, 'w') as dic:
        json.dump(vocab_dictionary, dic, ensure_ascii=False)


def get_train_validate_split(label_path, img_path, temp, logging):
    """ split the dataset into train and validate based the label_list """
    logging.info('The labes comes from {:s}'.format(label_path))
    logging.info('The images comes from {:s}'.format(img_path))
    label_train_test = [os.path.splitext(i)[0] for i in os.listdir(label_path)
                        if i.endswith('ls') or i.endswith('lst')]
    img_train_test = [os.path.splitext(i)[0] for i in os.listdir(img_path) if i.endswith('png')]
    train_test_names = list(set(label_train_test).intersection(set(img_train_test)))
    length = len(train_test_names)
    logging.info(
        'The total nums is [{:d}] for the intersection of the label and images'.format(length))
    train_nums = int(0.9 * length)
    test_nums = int(0.98 * length)
    logging.info('Train nums is [{:d}]; Test num is [{:d}]; and validate num is [{:d}]'.format(
        train_nums, test_nums - train_nums, length - test_nums))
    random.seed(10)
    random.shuffle(train_test_names)
    train_list = train_test_names[:train_nums]
    test_list = train_test_names[train_nums:test_nums]
    validate_list = train_test_names[test_nums:]
    with open(os.path.join(temp, 'train_name.lst'), 'w') as tr:
        tr.write('\n'.join(train_list))
    with open(os.path.join(temp, 'test_name.lst'), 'w') as te:
        te.write('\n'.join(test_list))
    with open(os.path.join(temp, 'validate_name.lst'), 'w') as te:
        te.write('\n'.join(validate_list))


def generate_numpy_data(
        voacbulary_path, label_augumentation_dir, img_augumentation_dir,
        prepared_dir, properties_json, temp_dir, logging):
    if not os.path.exists(
            os.path.join(temp_dir, 'train_name.lst')):
        logging.info('Split the dataset to train, test and validation...')
        get_train_validate_split(
            label_augumentation_dir, img_augumentation_dir, temp_dir, logging)
    logging.info('Split dataset done...')
    """ make the dataset with numpy format """
    vocab = ['<START>', '<EOS>', '<UNK>', '<PAD>'] + [i.strip()
                                                      for i in open(voacbulary_path).readlines()]

    str_to_idx = {x: i for i, x in enumerate(vocab)}
    idx_to_str = {value: key for key, value in str_to_idx.items()}
    properties = {}
    properties['vocab_size'] = len(vocab)
    properties['vocab'] = vocab
    properties['str_to_idx'] = str_to_idx
    properties['idx_to_str'] = idx_to_str
    for set in ['train', 'test', 'validate']:
        logging.info('Runing the [{}] dataset ...'.format(set))
        name_list = open(os.path.join(temp_dir, set + '_name.lst'), 'r').readlines()
        set_list, missing = [], {}
        logging.info('Quantity of samples is [{}]'.format(len(name_list)))
        for name in name_list:
            name = name.strip()
            if os.path.exists(
                    os.path.join(label_augumentation_dir, name + '.ls')):
                label_str = open(os.path.join(
                    label_augumentation_dir, name + '.ls')).readlines()
            else:
                label_str = open(os.path.join(
                    label_augumentation_dir, name + '.lst')).readlines()
            form = label_str[0].split()
            out_form = [str_to_idx['<START>']]
            for char in form:
                try:
                    out_form += [str_to_idx[char]]
                except:
                    if char not in missing.keys():
                        print(char, " not found!")
                        missing[char] = 0
                    missing[char] += 1
                    out_form += [str_to_idx['<UNK>']]
            out_form += [str_to_idx['<EOS>']]
            set_list.append([name, out_form])
        logging.info('Total dataset quantily exist : %d' % len(set_list))
        buckets = {}
        file_not_found_count = 0
        file_not_found = []
        for img_name, label in tqdm(set_list):
            img_path = os.path.join(img_augumentation_dir, img_name + '.png')
            if os.path.exists(img_path):
                img_shp = Image.open(img_path).size
                try:
                    if img_shp not in buckets:
                        buckets[img_shp] = [(img_name, label)]
                    buckets[img_shp] += [(img_name, label)]
                except Exception as e:
                    logging.error('Exception erros {}'.format(e))
            else:
                file_not_found_count += 1
                file_not_found.append(img_name)
        bucket_size = {str(key): len(buckets[key]) for key in buckets}

        properties['buck_size_nums'] = bucket_size

        if not os.path.exists(os.path.join(prepared_dir, 'properties.npy')):
            np.save(os.path.join(prepared_dir, 'properties'), properties)
            with open(properties_json, 'w') as js:
                json.dump(properties, js, ensure_ascii=False)
        logging.info('Save properties to {:s}'.format(os.path.join(prepared_dir, 'properties.npy')))
        logging.info(
            'Num files found in %s set: %d/%d' %
            (set, len(set_list) - file_not_found_count, len(set_list)))
        logging.info('Missing char: %s' % str(missing))
        logging.info('Missing files: %s' % str(file_not_found))
        logging.info('size_list: %s' % str(buckets.keys()))

        logging.info('Saving dataset as the numpy format ...')
        np.save(
            os.path.join(prepared_dir, set + '_buckets'), buckets)
        buckets_num = {}
        for i in buckets:
            buckets_num[i] = len(buckets[i])
        csv_columns = ['bucket_size', 'sample_nums']
        with open(os.path.join(prepared_dir, set + '_buckets.csv'), 'w') as jt:
            w = csv.writer(jt)
            w.writerow(csv_columns)
            w.writerows(buckets_num.items())


# DataGenerate = ImageDataGenerator(rotation_range=0.05,  # 图片随机旋转的角度
#                                   width_shift_range=0.01,  # 图片宽度的变化比例
#                                   height_shift_range=0,  # 图片高度的变化比例
#                                   shear_range=0.05,  # 逆时针方向的剪切变换角度
#                                   zoom_range=0.1,  # 随机缩放的幅度
#                                   zca_whitening=True,  # zca白噪声
#                                   zca_epsilon=1e-5,
#                                   horizontal_flip=False,  # 进行随机水平翻转
#                                   vertical_flip=False,  # 进行随机竖直方向的翻转
#                                   rescale=0.95,  # 重缩放因子
#                                   fill_mode='nearest',
#                                   data_format='channels_last')

# DataGenerate = ImageDataGenerator(rotation_range=1,  # 图片随机旋转的角度
#                                   width_shift_range=0.05,  # 图片宽度的变化比例
#                                   height_shift_range=0.1,  # 图片高度的变化比例
#                                   shear_range=0.1,  # 逆时针方向的剪切变换角度
#                                   zoom_range=0.1,  # 随机缩放的幅度
#                                   zca_whitening=True,  # zca白噪声
#                                   zca_epsilon=1e-2,
#                                   horizontal_flip=False,  # 进行随机水平翻转
#                                   vertical_flip=False,  # 进行随机竖直方向的翻转
#                                   rescale=0.95,  # 重缩放因子
#                                   fill_mode='nearest',
#                                   data_format='channels_last')


def img_augumentation(img_dir, label_dir, img_augumentation_dir,
                      label_augumentation_dir):
    """ image augumentation process

        NOTE: Modified the original code of keras
    """
    global DataGenerate
    import shutil
    img_list = [i for i in os.listdir(img_dir) if i.endswith('.png')]
    print('Total imgs to augument:', len(img_list))
    count = 1
    for img_name in img_list:
        print(count, '/', len(img_list))
        count += 1
        img_path = os.path.join(img_dir, img_name)
        new_img_path = os.path.join(img_augumentation_dir, img_name)
        shutil.copy(img_path, new_img_path)

        file_name, file_ext = os.path.splitext(img_name)
        if not os.path.exists(os.path.join(label_dir, file_name + '.lst')):
            label_path = os.path.join(label_dir, file_name + '.ls')
        else:
            label_path = os.path.join(label_dir, file_name + '.lst')
        shutil.copy(label_path, os.path.join(
            label_augumentation_dir, os.path.basename(label_path)))

        img = load_img(img_path)
        # This is a Numpy array with shape (3, 150, 150)
        img_np = img_to_array(img)
        # This is a Numpy array with shape (1, 3, 150, 150)
        img_np = img_np.reshape((1,) + img_np.shape)
        """ 此处对flow源代码进行了修改，使函数返回文件名 """
        imageGen = DataGenerate.flow(
            img_np, batch_size=1, save_to_dir=img_augumentation_dir,
            save_prefix=file_name, save_format=file_ext.split('.')[-1])
        idx = 0

        for img_enhance_name in imageGen:
            new_filename = img_enhance_name.split('.')[0]
            new_img_path = os.path.join(img_augumentation_dir, img_enhance_name)
            new_label_path = os.path.join(
                label_augumentation_dir, new_filename + os.path.splitext(label_path)[-1])
            try:
                shutil.copy(label_path, new_label_path)

            except Exception as e:

                print('error is:', e)
            idx += 1
            if idx == 5:
                break  # otherwise the generator would loop indefinitely

# padding_img_path = '/home/xiaofeng/data/char_formula/enhance_different_thresh/img_padding'
# img_augumentation_dir = '/home/xiaofeng/data/char_formula/enhance_different_thresh/img_augumentation_backup'
# label_augumentation_dir = '/home/xiaofeng/data/char_formula/enhance_different_thresh/label_augumentation_backup'
# label_path = '/home/xiaofeng/data/char_formula/enhance_different_thresh/label_ori'
# img_augumentation(padding_img_path, label_path,
#                   img_augumentation_dir, label_augumentation_dir)


def dataset_format_exchange(
        dataset, ori_label_dir, ori_img_dir, new_img_dir, new_label_dir):
    """ 
    This is a temp process
    make the original dataset format to the nuiform
    """
    if not os.path.exists(new_label_dir):
        os.makedirs(new_label_dir)
    if not os.path.exists(new_img_dir):
        os.makedirs(new_img_dir)
    datasset_list = open(dataset).readlines()
    label_list = open(ori_label_dir).readlines()
    label_filter = []
    print('length:', len(datasset_list))
    cur = 0
    for img_name_idx in datasset_list:
        try:
            if cur % 10000 == 0:
                print(cur, '/', len(datasset_list))
            cur += 1
            label_idx, img_name, _ = img_name_idx.split(' ')
            # name, ext = os.path.splitext(img_name)
            old_img_path = os.path.join(ori_img_dir, img_name+'.png')
            label = label_list[int(label_idx)]
            if label in label_filter:
                continue
            if label not in label_filter:
                label_filter.append(label)
                new_label_path = os.path.join(new_label_dir, img_name + '.lst')
                new_img_path = os.path.join(new_img_dir, img_name+'.png')
                shutil.copy(old_img_path, new_img_path)
                with open(new_label_path, 'w') as la:
                    la.write(label)
        except Exception as e:
            print('error:', str(e))
            pass


# dataset = './formula/dataset/data_label/dataset.lst'
# ori_label_dir = './formula/dataset/data_label/formula_normal.lst'
# ori_img_dir = '/home/xiaofeng/data/formula/origin/img_ori'
# new_img_dir = '/home/xiaofeng/data/formula/origin/img_filtered'
# new_label_dir = '/home/xiaofeng/data/formula/origin/label_ori'
# dataset_format_exchange(dataset, ori_label_dir, ori_img_dir, new_img_dir, new_label_dir)


def split_different_aug(
    temp_dir, label_augumentation_dir, img_augumentation_dir,
        aug_img_backup, aug_label_backup):
    """ 
    This is a temp process aimed to move the img and label enhanced to the backup dir.
    """
    for set in ['train', 'validate']:
        print('Moving aug img to the backup based {} names'.format(set))
        name_list = open(
            os.path.join(temp_dir, set + '_name_1_aug.ls'),
            'r').readlines()
        print(
            'There contains {} imgs of the {} dataset'.format(
                len(name_list),
                set))
        count = 1
        for name in name_list:
            try:
                print('{}/{}'.format(count, len(name_list)))
                name = name.strip()
                if os.path.exists(
                        os.path.join(label_augumentation_dir, name + '.ls')):
                    label_path = os.path.join(
                        label_augumentation_dir, name + '.ls')
                else:
                    label_path = os.path.join(
                        label_augumentation_dir, name + '.lst')

                new_label_path = os.path.join(
                    aug_label_backup, os.path.basename(label_path))
                shutil.copy(label_path, new_label_path)
                img_path = os.path.join(img_augumentation_dir, name + '.png')
                new_img_path = os.path.join(aug_img_backup, name + '.png')
                shutil.copy(img_path, new_img_path)
                count += 1
            except Exception as e:
                print('error:', e)


# temp_dir = './char_formula/enhance_different_threshold/temp'
# img_augumentation_dir = '/home/xiaofeng/data/char_formula/enhance_different_thresh/img_augumentation'
# label_augumentation_dir = '/home/xiaofeng/data/char_formula/enhance_different_thresh/label_augumentation'
# aug_img_backup = '/home/xiaofeng/data/char_formula/enhance_different_thresh/img_augumentation_backup'
# aug_label_backup = '/home/xiaofeng/data/char_formula/enhance_different_thresh/label_augumentation_backup'

# if not os.path.exists(aug_img_backup):
#     os.makedirs(aug_img_backup)
# if not os.path.exists(aug_label_backup):
#     os.makedirs(aug_label_backup)

# split_different_aug(temp_dir, label_augumentation_dir,
#                     img_augumentation_dir, aug_img_backup, aug_label_backup)
