'''
Filename: dataset_iter.py
Project: image2katex
File Created: Wednesday, 5th December 2018 3:24:14 pm
Author: xiaofeng (sxf1052566766@163.com)
--------------------------
Last Modified: Sunday, 9th December 2018 4:38:25 pm
Modified By: xiaofeng (sxf1052566766@163.com)
---------------------------
Copyright: 2018.06 - 2018 OnionMath. OnionMath
'''

import os
import random
from math import ceil
import cv2
from utils.process_image import img_aug
import numpy as np
from PIL import Image


class DataIteratorSeq2SeqAtt(object):
    def __init__(self, _config, _logging, data_set, **kwargs):
        self._config = _config
        self._logging = _logging
        self._image_folder = self._config.dataset.get('image_folder')
        self.pad_idx = self._config.dataset.get("id_pad")
        self.end_idx = self._config.dataset.get("id_end")
        self._prepared_dir = self._config.dataset.get('prepared_folder')
        self._set = data_set
        assert isinstance(self._set, list), 'Input dataset mus be list'
        assert isinstance(self._prepared_dir, list) and isinstance(
            self._image_folder, list), 'Input dataset details must be list format'

    def generate(self, _seed=100):
        """ Generate iters """
        for _dataset in self._set:
            if _dataset in ['train', 'validate']:
                batch_size = self._config.model.batch_size
            else:
                assert _dataset in ['test'], '_dataset must be [train,validate,test]'
                batch_size = self._config.model.test_batch_size
            self._logging.info('Generate the [{}] dataset'.format(_dataset))
            if len(self._prepared_dir) > 1:
                # because the seed can be used for once, so set seed for each list
                random.seed(_seed)
                random.shuffle(self._prepared_dir)
                random.seed(_seed)
                random.shuffle(self._image_folder)
            for idx in range(len(self._prepared_dir)):
                _dataset_dir = os.path.join(self._prepared_dir[idx], _dataset + '_buckets.npy')
                _image_folder = self._image_folder[idx]
                self._logging.info('Load dataset is [{:s}]'.format(_image_folder))
                dataset_details = np.load(_dataset_dir).tolist()
                bucket_size = [x for x in dataset_details if x[0]*x[1] < 48000]
                # bucket_size = [x for x in dataset_details]
                # bucket_size=[x for x in dataset_details]
                random.shuffle(bucket_size)
                total_nums = sum((len(dataset_details[x]) for x in dataset_details))
                self._logging.info('Total num is [{:d}]'.format(total_nums))
                for bucket in bucket_size:
                    bucket_details = dataset_details[bucket]
                    set_list = [(image_name, label) for image_name, label in bucket_details
                                if os.path.exists(os.path.join(_image_folder, image_name + '.png'))]
                    random.shuffle(set_list)
                    dataset_num = len(set_list)
                    if _dataset in ['train']:
                        if dataset_num < 100:
                            continue
                    _iters = int(ceil(dataset_num / batch_size))
                    self._logging.info('Total iter of the bucket size [{}] is {}'.format(
                        bucket, _iters))
                    for i in range(_iters):
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, dataset_num-1)
                        _samples_nums = end_idx - start_idx
                        if _samples_nums != batch_size:
                            continue
                        _sublist = set_list[start_idx:end_idx]
                        batch_imgs = []
                        _batch_forms = []
                        batch_names = []
                        for image_name, label in _sublist:
                            image_path = os.path.join(_image_folder, image_name + '.png')
                            rand = random.random()
                            # process image
                            if np.random.rand() < 0.5:
                                img = cv2.imread(image_path)
                                im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                image_data = cv2.adaptiveThreshold(
                                    im_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10)
                            else:
                                image_data = np.asarray(Image.open(image_path).convert('L'))

                            # image  augment
                            if self._config.datatype == 'Augment':
                                if np.random.rand() > 0.5:
                                    image_data = img_aug(img_path=image_path)
                                else:
                                    image_data = np.asarray(Image.open(image_path).convert('L'))
                            if image_data.ndim == 2:
                                image_data = image_data[:, :, np.newaxis]
                            batch_imgs.append(image_data)
                            _batch_forms.append(label)
                            batch_names.append(image_path)

                        if _dataset in ['test']:
                            assert len(batch_imgs) == len(
                                _batch_forms) == 1, 'In "test" model batch must be one'

                            yield batch_imgs[0], _batch_forms[0]

                        else:
                            max_len = max(map(lambda x: len(x), _batch_forms))
                            batch_formulas = self.pad_idx * np.ones([len(_batch_forms),
                                                                     max_len + 1],
                                                                    dtype=np.int32)
                            batch_formula_length = np.zeros(len(_batch_forms), dtype=np.int32)

                            for idx, formula in enumerate(_batch_forms):
                                # now the formulua sequence is [start,...,end]
                                batch_formulas[idx, : len(formula)] = formula
                                # and the input is [start ,....]
                                # the target is [....,end]
                                # padding sequence is [start,...,end,pad,...]
                                batch_formula_length[idx] = len(formula) - 1
                            yield batch_imgs, batch_formulas, batch_formula_length, batch_names


class DataIteratorErrorChecker(object):
    """ Dataloade for the ErrorChecker Model """

    def __init__(self, _config, _logging, data_set, **kwargs):
        self._config = _config
        self._logging = _logging
        self._bucket_size = self._config.dataset.get("bucket_size")
        self.pad_idx = self._config.dataset.get("id_pad")
        self._prepared_dir = self._config.dataset.get('prepared_folder')
        self._set = data_set
        assert isinstance(self._set, list), 'Input dataset mus be list'
        assert isinstance(self._prepared_dir, list), 'Input dataset details must be list format'

    def generate(self, _seed=100):
        """ Generate iters """
        for _dataset in self._set:
            if _dataset in ['train', 'validate']:
                batch_size = self._config.model.batch_size
            else:
                assert _dataset in ['test'], '_dataset must be [train,validate,test]'
                batch_size = self._config.model.test_batch_size
            self._logging.info('Generate the [{}] dataset'.format(_dataset))
            if len(self._prepared_dir) > 1:
                # because the seed can be used for once, so set seed for each list
                random.seed(_seed)
                random.shuffle(self._prepared_dir)
            for idx in range(len(self._prepared_dir)):
                _dataset_dir = os.path.join(self._prepared_dir[idx], _dataset + '_buckets.npy')
                self._logging.info('Load the dataset: {:s}'.format(_dataset_dir))
                dataset_details = np.load(_dataset_dir).tolist()
                bucket_size = self._bucket_size
                random.shuffle(bucket_size)
                total_nums = sum((len(dataset_details[x]) for x in dataset_details))
                self._logging.info('Total num is [{:d}]'.format(total_nums))
                self._logging.info('bucket size is: {}'.format(bucket_size))
                for bucket in bucket_size:
                    bucket_details = dataset_details[bucket]
                    random.shuffle(bucket_details)
                    dataset_num = len(bucket_details)
                    _iters = int(ceil(dataset_num / batch_size))
                    self._logging.info('Total iter of the bucket size [{}]is {}'.format(
                        bucket, _iters))
                    source_size, traget_size = bucket
                    for i in range(_iters):
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, dataset_num-1)
                        _samples_nums = end_idx - start_idx
                        if _samples_nums != batch_size:
                            continue
                        _sublist = bucket_details[start_idx:end_idx]
                        batch_source = []
                        batch_traget = []
                        for sources_seq, target_seq in _sublist:
                            batch_source.append(sources_seq)
                            batch_traget.append(target_seq)
                        if _dataset in ['test']:
                            assert len(batch_source) == len(
                                batch_traget) == 1, 'In "test" model batch must be one'
                            yield batch_source[0], batch_traget[0]

                        else:
                            batch_source_pad = self.pad_idx * np.ones([len(batch_source),
                                                                       source_size],
                                                                      dtype=np.int32)
                            batch_traget_pad = self.pad_idx * np.ones([len(batch_source),
                                                                       traget_size],
                                                                      dtype=np.int32)

                            batch_target_length = np.zeros(len(batch_traget), dtype=np.int32)
                            for idx, tar_seq in enumerate(batch_traget):
                                # now the target sequence is [start,...,end]
                                batch_traget_pad[idx, :len(tar_seq)] = tar_seq
                                # the input is [start ,....]
                                # the label is [....,end]
                                # padding sequence is [start,...,end,pad,...]
                                batch_target_length[idx] = len(tar_seq) - 1
                            for idx, source_seq in enumerate(batch_source):
                                batch_source_pad[idx, :len(source_seq)] = source_seq
                            yield batch_source_pad, batch_traget_pad, batch_target_length


class DataIteratorDis(object):
    """ Define the iterator for the Discriminator """
    def __init__(self, _config, _logging, data_set, **kwargs):
        self._config = _config
        self._logging = _logging
        self._bucket_size = self._config.dataset.get("bucket_size")
        self.pad_idx = self._config.dataset.get("id_pad")
        self._prepared_dir = self._config.dataset.get('prepared_folder')
        self._set = data_set
        assert isinstance(self._set, list), 'Input dataset mus be list'
        assert isinstance(self._prepared_dir, list), 'Input dataset details must be list format'
        
    def generate(self,):

