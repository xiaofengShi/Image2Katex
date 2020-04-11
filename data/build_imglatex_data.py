'''
File: data_processing.py
Project: dataset
File Created: Wednesday, 11th July 2018 6:33:11 pm
Author: xiaofeng (sxf1052566766@163.com)
-----
Last Modified: Wednesday, 11th July 2018 6:33:16 pm
Modified By: xiaofeng (sxf1052566766@163.com>)
-----
 2018.06 - 2018 Latex Math, Latex Math
'''

import json
import logging
import os
import pickle
import shutil
from multiprocessing import Pool

from config_dataset import Config as cfg
from data_utils import *

THREAD = 4
AUG = True
"""
用于生成训练用文件,
    1. 对原始图片进行crop，对截取的区域小于100的，文字为多行的去掉，截取文字部分的最小矩形边框
    2. croop之后获取图片的尺寸列表
    3. 使用聚类的方法将图片尺寸归纳为k类
    4. 使用聚类之后的尺寸对图片进行resize
    5. 创建训练数据集
问题：聚类之后的图片尺寸仍然很大，修改resize的方法:
        将图片的尺寸进行高度方向的限制，设置三种尺寸(10,20,30,40)，保持高度和宽度比例不变进行图片修正，
        这样之后，图片的尺寸高度进行了固定可是长度方向变化很大
        之后在进行图像尺寸的聚类，得到k类图像宽度的分布，对不满足的图像尺寸进行长度方向的padding，
        最终确定图片的bucket
"""


def initiate_dir(dir_name):
    """Creates directory if it does not exists"""
    if dir_name is not None:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            return True
        else:
            return False


def generate_dataset(parameters, config):
    """ generate the train dataset """
    img_ori, train_list_dir, test_list_dir, validate_list_dir, croped_dir, size_first_resized, \
        size_ori_dict, img_augumentation_dir, label_augumentation_dir, resized_size_bucket, \
        resize_img_path, padding_img_path, croped_img_names, voacbulary_path, vocab_dict_path, \
        properties_json, label_path, formula_file, temp_dir, prepared_dir = parameters
    # assert os.path.exists(img_ori), '{} does not exist'.format(img_ori)
    THREAD = config.thread_nums
    if not os.path.exists(prepared_dir):
        os.makedirs(prepared_dir)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    """ crop the ori img """
    assert os.path.exists(img_ori) and os.path.exists(formula_file)
    ori_filelist = [i for i in os.listdir(img_ori) if i.endswith('.png')]
    ori_label_list = open(formula_file).readlines()

    assert len(ori_filelist) <= len(
        ori_label_list), 'Original image nums and formula nums must be same... '

    if not os.path.exists(label_path):
        os.makedirs(label_path)

    label_folder_list = [i for i in os.listdir(label_path) if i.endswith('.lst')]

    if config.process_original and len(label_folder_list) <= int(0.95 * len(ori_label_list)):

        total = [train_list_dir, test_list_dir, validate_list_dir]
        out = []
        for set_dir in total:
            logging.info('Runing the {:s}'.format(set_dir))
            set_list = open(set_dir).readlines()
            for line in set_list:
                formula_idx, image_name, _ = line.strip().split()
                formula = ori_label_list[int(formula_idx)]
                formula = formula.strip()
                image = image_name+'.png'
                out.append([image, formula])
                with open(os.path.join(label_path, image_name + '.lst'), 'w') as fo:
                    fo.write('%s\n' % formula)

        with open(os.path.join(temp_dir, 'im2latex_datset.pkl'), 'wb') as fi:
            pickle.dump(out, fi, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info('Preprocess the original dataset and generate the label files ')

    if config.croped_flage:
        croped_actual_flage = initiate_dir(croped_dir)
        if not croped_actual_flage:
            croped_filelist = [i for i in os.listdir(croped_dir) if i.endswith('.png')]
            if len(croped_filelist) > int(0.8*len(ori_filelist)):
                logging.info(
                    'Croped folder is exist and the image numm equal with original image folder')
                logging.info('Skip croping task ...')
        else:
            try:
                shutil.rmtree(croped_dir)
            except:
                pass
            os.makedirs(croped_dir)
            logging.info(
                'Image num is {:d} and will be saved to the folder {:s}'.format(
                    len(ori_filelist),
                    croped_dir))
            logging.info('Creating pool with %d threads for the croping task ...' % THREAD)
            pool = Pool(THREAD)
            logging.info('Croping image task running...')
            croped_names = list(
                pool.map(
                    crop_image,
                    [(os.path.join(img_ori, filename), croped_dir, filename) for filename in ori_filelist]))
            logging.info('Save out image name to {:s}'.format(croped_img_names))
            logging.info('Croped success image num is [{:d}]'.format(len(croped_names)))
            with open(croped_img_names, 'wb') as ou:
                pickle.dump(croped_names, ou, pickle.HIGHEST_PROTOCOL)
            pool.close()
            pool.join()
            logging.info('Croping image done and go to the next task...')
            del croped_names
    del ori_filelist

    if config.resize_flage:
        if not os.path.exists(size_ori_dict):
            logging.info('Get the image dataset details ...')
            logging.info('Details saved to the folder {}'.format(size_ori_dict))
            get_img_size_dict(croped_dir, size_ori_dict)
        with open(size_ori_dict, 'r') as dic:
            size_dict = json.load(dic)
            logging.info('Load size dictionary done ...')
        height_croped = size_dict['height']
        resized_actual_flage = initiate_dir(resize_img_path)

        croped_names = [i for i in os.listdir(croped_dir) if i.endswith('.png')]

        if not resized_actual_flage:
            resized_names = [i for i in os.listdir(resize_img_path) if i.endswith('.png')]
            if len(resized_names) >= int(0.9 * len(croped_names)):
                logging.info('Resized folder is exist and the image numm equal with croped image nums')
                logging.info('Skip resizing task ...')
                del resized_names
        else:
            shutil.rmtree(resize_img_path)
            os.makedirs(resize_img_path)
            logging.info('Resizing  pool with %d threads for the croping task ...' % THREAD)
            logging.info('Resize image task runing ...')
            pool = Pool(THREAD)
            pool.map(
                resize_img_target_height,
                [(filename, croped_dir, height_croped, resize_img_path, config.height_list)
                 for filename in croped_names])
            pool.close()
            pool.join()
            logging.info('Resize image done and go to the next task ...')
        del croped_names

    """ padding the resized img based resized_size_bucket """
    if config.padding_flage:
        """ get the size bucket of the resized img """
        if not os.path.exists(resized_size_bucket):
            logging.info('K_means to  get the size buckets of the resized imgs...')
            size_bucket_list = k_means_size_list(
                resize_img_path, size_first_resized, resized_size_bucket, config.height_list)
            logging.info('Generate size buckets done ...')
        else:
            with open(resized_size_bucket, 'rb') as size:
                size_bucket_list = pickle.load(size)
        logging.info('Size bucket is {}'.format(size_bucket_list))

        padding_actual_flage = initiate_dir(padding_img_path)
        resized_names = [i for i in os.listdir(resize_img_path) if i.endswith('.png')]
        if not padding_actual_flage:
            padding_names = [i for i in os.listdir(padding_img_path) if i.endswith('.png')]
            if len(padding_names) >= int(0.9*len(resized_names)):
                logging.info('Resized folder is exist and the image numm equal with croped image nums')
                logging.info('Skip padding task ...')
                del padding_names
        else:
            shutil.rmtree(padding_img_path)
            os.makedirs(padding_img_path)
            logging.info('Padding  pool with %d threads for the padding task ...' % THREAD)
            logging.info('Padding image task runing ...')
            pool = Pool(THREAD)
            pool.map(img_padding, [(filename, resize_img_path, size_bucket_list, padding_img_path)
                                   for filename in resized_names])
            pool.close()
            pool.join()
            logging.info('Padding image done and go to the next task ...')
        del resized_names

    """ img augumentation """
    if config.augumentation_flag:

        try:
            os.makedirs(img_augumentation_dir)
            os.makedirs(label_augumentation_dir)
        except:
            pass
        # img_augumentation(
        #     padding_img_path, label_path, img_augumentation_dir,
        #     label_augumentation_dir)
        image_aug(
            padding_img_path, label_path, img_augumentation_dir,
            label_augumentation_dir)
        print('Step5: img augumentation done')
        dataset_img = img_augumentation_dir
        dataset_label = label_augumentation_dir
    else:
        logging.info('Skipping the augumentation process and generate the uniform dataset')
        dataset_img = padding_img_path
        dataset_label = label_path

    if not os.path.exists(voacbulary_path):
        parameters_generate_vacab = dataset_label, voacbulary_path, vocab_dict_path, temp_dir, logging
        get_vocabluary(parameters_generate_vacab)
        logging.info('Get vocabulary form the label files ...')
        logging.info('Go to the next step ...')

    print('Make the dataset with numpy format')
    logging.info('Generate the dataset as the numpy dataset ... ')
    generate_numpy_data(voacbulary_path, dataset_label,
                        dataset_img, prepared_dir, properties_json, temp_dir, logging)
    print('Final: dataset has made')


if __name__ == '__main__':
    print('local or remote represents 0 or 1')
    print('Enhance style is: ori, diff_threshold, Latex')

    local_or_not = int(input('Please select the location: local(0) or remote(1):'))
    enhance_type = int(
        input(
            'Please select enhance style: ori(0) ,handwritten(1):'))
    dataset_config = cfg(local_falge=local_or_not, enhance_flage=enhance_type)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename='dataset_log.log')

    logging.info('*'*50)
    logging.info('Generating the dataset ....')
    logging.info('local is {} and enhance is {}'.format(local_or_not, enhance_type))

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info('Script being executed: %s' % __file__)

    parameters = dataset_config.dir_settings

    generate_dataset(parameters, dataset_config)
    logging.info('All Done ...')
    logging.info('#'*50)

    print('done')
