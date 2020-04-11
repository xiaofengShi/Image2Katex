'''
Filename: config_dataset.py
Project: data
File Created: Sunday, 2nd December 2018 10:51:53 pm
Author: xiaofeng (sxf1052566766@163.com)
--------------------------
Last Modified: Tuesday, 11th December 2018 10:58:47 am
Modified By: xiaofeng (sxf1052566766@163.com)
---------------------------
: 2018.06 - 2018 . 
'''
import numpy as np
import os


class Config(object):
    def __init__(self, local_falge, enhance_flage):
        self.local_flage = local_falge
        self.enhance_flage = enhance_flage
        self.thread_nums = 13
        self.process_original = True
        self.croped_flage = True
        self.resize_flage = True
        self.padding_flage = True
        self.augumentation_flag = False
        self.height_list = [24, 32, 40, 48, 64, 80, 100, 120, 200, 320]
        self.setup()

    def setup(self):
        """ setup dirs for the different dataset """

        if self.local_flage == 0 and self.enhance_flage == 0:
            """ local original dataset """
            img_ori = '/Users/xiaofeng/Desktop/ori/formula_images'
            train_list_dir = '/Users/xiaofeng/Desktop/ori/im2latex_train.lst'
            test_list_dir = '/Users/xiaofeng/Desktop/ori/im2latex_test.lst'
            validate_list_dir = '/Users/xiaofeng/Desktop/ori/im2latex_validate.lst'
            croped_dir = '/Users/xiaofeng/Desktop/ori/process/img_croped'
            resize_img_path = '/Users/xiaofeng/Desktop/ori/process/img_resized'
            padding_img_path = '/Users/xiaofeng/Desktop/ori/process/img_padding'
            img_augumentation_dir = '/Users/xiaofeng/Desktop/ori/process/img_augumentation'
            label_augumentation_dir = '/Users/xiaofeng/Desktop/ori/process/label_augumentation'
            label_path = '/Users/xiaofeng/Desktop/ori/label_ori'
            formula_file = '/Users/xiaofeng/Desktop/ori/process/im2latex_formulas_normal.lst'
            temp_dir = '/Users/xiaofeng/Desktop/ori/temp'
            prepared_dir = '/Users/xiaofeng/Desktop/ori/process/prepared'
            size_first_resized = temp_dir+'/resize_img_size.pkl'
            size_ori_dict = temp_dir+'/size_ori_sorted.json'
            resized_size_bucket = temp_dir+'/resized_size_bucket.pkl'
            croped_img_names = temp_dir+'/croped_img_names.pkl'
            voacbulary_path = prepared_dir+'/vocabulary.txt'
            vocab_dict_path = temp_dir + '/vocabulary.json'
            properties_json = temp_dir + '/properties.json'

        elif self.local_flage == 1 and self.enhance_flage == 0:
            """ remote original dataset """
            img_ori = '/home/xiaofeng/data/image2latex/original/original_dataset/formula_images'
            train_list_dir = '/home/xiaofeng/data/image2latex/original/original_dataset/im2latex_train.lst'
            test_list_dir = '/home/xiaofeng/data/image2latex/original/original_dataset/im2latex_test.lst'
            validate_list_dir = '/home/xiaofeng/data/image2latex/original/original_dataset/im2latex_validate.lst'
            croped_dir = '/home/xiaofeng/data/image2latex/original/process/img_croped'
            resize_img_path = '/home/xiaofeng/data/image2latex/original/process/img_resized'
            padding_img_path = '/home/xiaofeng/data/image2latex/original/process/img_padding'
            img_augumentation_dir = '/home/xiaofeng/data/image2latex/original/process/img_augumentation'
            label_augumentation_dir = '/home/xiaofeng/data/image2latex/original/process/label_augumentation'
            label_path = '/home/xiaofeng/data/image2latex/original/process/label_ori'
            formula_file = '/home/xiaofeng/data/image2latex/original/process/im2latex_formulas_normal.lst'
            temp_dir = './im2latex_dataset/original/temp'
            prepared_dir = './im2latex_dataset/original/prepared'
            size_first_resized = temp_dir+'/resize_img_size.pkl'
            size_ori_dict = temp_dir + '/size_ori_sorted.json'
            resized_size_bucket = temp_dir+'/resized_size_bucket.pkl'
            croped_img_names = temp_dir+'/croped_img_names.pkl'
            voacbulary_path = prepared_dir+'/vocabulary.txt'
            vocab_dict_path = temp_dir+'/vocabulary.json'
            properties_json = temp_dir+'/properties.json'

        elif self.local_flage == 0 and self.enhance_flage == 1:
            """ local handwritten dataset """
            img_ori = '/Users/xiaofeng/Desktop/handwritten/images'
            train_list_dir = '/Users/xiaofeng/Desktop/handwritten/train.lst'
            test_list_dir = '/Users/xiaofeng/Desktop/handwritten/test.lst'
            validate_list_dir = '/Users/xiaofeng/Desktop/handwritten/val.lst'
            croped_dir = '/Users/xiaofeng/Desktop/handwritten/process/img_croped'
            resize_img_path = '/Users/xiaofeng/Desktop/handwritten/process/img_resized'
            padding_img_path = '/Users/xiaofeng/Desktop/handwritten/process/img_padding'
            img_augumentation_dir = '/Users/xiaofeng/Desktop/handwritten/process/img_augumentation'
            label_augumentation_dir = '/Users/xiaofeng/Desktop/handwritten/process/label_augumentation'
            label_path = '/Users/xiaofeng/Desktop/handwritten/process/label_ori'
            formula_file = '/Users/xiaofeng/Desktop/handwritten/process/im2latex_formulas_normal.lst'
            temp_dir = '/Users/xiaofeng/Desktop/handwritten/temp'
            prepared_dir = '/Users/xiaofeng/Desktop/handwritten/process/prepared'
            size_first_resized = temp_dir + '/resize_img_size.pkl'
            size_ori_dict = temp_dir + '/size_ori_sorted.json'
            resized_size_bucket = temp_dir+'/resized_size_bucket.pkl'
            croped_img_names = temp_dir+'/croped_img_names.pkl'
            voacbulary_path = prepared_dir+'/vocabulary.txt'
            vocab_dict_path = temp_dir + '/vocabulary.json'
            properties_json = temp_dir + '/properties.json'

        elif self.local_flage == 1 and self.enhance_flage == 1:
            """ 
            Device: server; 
            dataset type: handwritten 
            """
            img_ori = '/home/xiaofeng/data/image2latex/handwritten/original_dataset/images'
            train_list_dir = '/home/xiaofeng/data/image2latex/handwritten/original_dataset/train.lst'
            test_list_dir = '/home/xiaofeng/data/image2latex/handwritten/original_dataset/test.lst'
            validate_list_dir = '/home/xiaofeng/data/image2latex/handwritten/original_dataset/val.lst'
            croped_dir = '/home/xiaofeng/data/image2latex/handwritten/process/img_croped'
            resize_img_path = '/home/xiaofeng/data/image2latex/handwritten/process/img_resized'
            padding_img_path = '/home/xiaofeng/data/image2latex/handwritten/process/img_padding'
            img_augumentation_dir = '/home/xiaofeng/data/image2latex/handwritten/process/img_augumentation'
            label_augumentation_dir = '/home/xiaofeng/data/image2latex/handwritten/process/label_augumentation'
            label_path = '/home/xiaofeng/data/image2latex/handwritten/process/label_ori'
            formula_file = '/home/xiaofeng/data/image2latex/handwritten/process/im2latex_formulas_normal.lst'
            temp_dir = './im2latex_dataset/handwritten/temp'
            prepared_dir = './im2latex_dataset/handwritten/prepared'
            size_first_resized = temp_dir+'/resize_img_size.pkl'
            size_ori_dict = temp_dir+'/size_ori_sorted.json'
            resized_size_bucket = temp_dir + '/resized_size_bucket.pkl'
            croped_img_names = temp_dir+'/croped_img_names.pkl'
            voacbulary_path = prepared_dir+'/vocabulary.txt'
            vocab_dict_path = temp_dir+'/vocabulary.json'
            properties_json = temp_dir+'/properties.json'

        self.dir_settings = (
            img_ori, train_list_dir, test_list_dir, validate_list_dir, croped_dir,
            size_first_resized, size_ori_dict, img_augumentation_dir,
            label_augumentation_dir, resized_size_bucket, resize_img_path,
            padding_img_path, croped_img_names, voacbulary_path, vocab_dict_path, properties_json,
            label_path, formula_file, temp_dir, prepared_dir)


class SequenceVocabulary:

    """  
    ['<START>', '<EOS>', '<UNK>', '<PAD>']=[0,1,2,3]
    """
    START_ID = 0
    EOS_ID = 1
    UNK_ID = 2
    PAD_ID = 3
    vocab_dir = os.path.abspath('./errorchecker_dataset/prepared/properties.npy')
    vocabulary = np.load(vocab_dir).tolist()
    vocab_size = vocabulary['vocab_size']
    idx_to_token = vocabulary['idx_to_str']
    token_to_idx = vocabulary['str_to_idx']
    bucket_size = [(22, 21), (33, 33), (42, 42), (52, 52), (61, 62), (72, 73),
                   (74, 18), (84, 86), (99, 101), (117, 121), (143, 147), (179, 192), (198, 58)]
