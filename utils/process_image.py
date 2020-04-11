'''
File: image.py
Project: utils
File Created: Wednesday, 28th November 2018 4:14:46 pm
Author: xiaofeng (sxf1052566766@163.com)
-----
Last Modified: Saturday, 22nd December 2018 11:59:35 am
Modified By: xiaofeng (sxf1052566766@163.com>)
-----
 2018.06 - 2018 Latex Math, Latex Math
'''

import os
from multiprocessing import Pool

import numpy as np
from PIL import Image
from numpy import random
import cv2
from utils.general import IMG_EXTENSIONS, delete_file, get_files, init_dir, run
import scipy.misc
TIMEOUT = 10


def get_max_shape(arrays):
    """
    Args:
        images: list of arrays

    """
    shapes = map(lambda x: list(x.shape), arrays)
    ndim = len(arrays[0].shape)
    max_shape = []
    for d in range(ndim):
        max_shape += [max(shapes, key=lambda x: x[d])[d]]

    return max_shape


def pad_batch_images(images, max_shape=None):
    """
    Args:
        images: list of arrays
        target_shape: shape at which we want to pad

    """

    # 1. max shape
    if max_shape is None:
        max_shape = get_max_shape(images)

    # 2. apply formating
    batch_images = 255 * np.ones([len(images)] + list(max_shape))
    for idx, img in enumerate(images):
        batch_images[idx, :img.shape[0], :img.shape[1]] = img

    return batch_images.astype(np.uint8)


def greyscale(state):
    """Preprocess state (:, :, 3) image into greyscale"""
    state = state[:, :, 0]*0.299 + state[:, :, 1]*0.587 + state[:, :, 2]*0.114
    state = state[:, :, np.newaxis]
    return state.astype(np.uint8)


def downsample(state):
    """Downsamples an image on the first 2 dimensions

    Args:
        state: (np array) with 3 dimensions

    """
    return state[::2, ::2, :]


def padding_img(parameters):
    filename, img_path, size_bucket, padding_dir = parameters
    try:
        old_im = Image.open(img_path)
        width, height = old_im.size
        # get the nearest size in the bucket given as the target
        width_list = [
            [size_bucket[i][0] - width, size_bucket[i]]
            for i in range(len(size_bucket))
            if size_bucket[i][1] == height and abs(size_bucket[i][0] - width) <= 50]
        if width_list:
            temp = [width_select[0] for width_select in width_list]
            target_idx = temp.index(min(temp))
            target_size = width_list[target_idx][1]
            new_im = Image.new("RGB", target_size, (255, 255, 255))
            new_im.paste(old_im)
            new_im.save(os.path.join(padding_dir, filename))
        return True
    except:
        return False


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
            # raise ValueError("cannot parse this case...")
            pass
    return peek_ranges


def crop_image(l):
    """ use Image to crop the image """
    img, crop_saved_path, file_name, _logger = l
    try:
        old_im = Image.open(img).convert('L')
        witdth, height = old_im.size
        img_data = np.asarray(old_im, dtype=np.uint8)  # height, width
        nnz_inds = np.where(img_data != 255)
        if len(nnz_inds[0]) == 0:
            _logger.info('There is no details found in the file [{}]'.format(img))
            return False
        y_min = np.min(nnz_inds[0])
        y_max = np.max(nnz_inds[0])
        x_min = np.min(nnz_inds[1])
        x_max = np.max(nnz_inds[1])
        if (x_max - x_min) * (y_max - y_min) < 100:
            _logger.info('The content found is too small in the file: [{}]'.format(img))
            return False
        old_im = old_im.crop((x_min, y_min, x_max + 1, y_max + 1))
        # NOTE: 查看截取的图像是否具有多行
        # croped_height = y_max - y_min
        # old_img_numpy = np.asarray(old_im, dtype=np.uint8)
        # _, adaptive_threshold = cv2.threshold(
        #     old_img_numpy, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # horizontal_sum_croped = np.sum(adaptive_threshold, axis=1)
        # peek_ranges_croped = extract_peek_ranges_from_array(
        #     horizontal_sum_croped, minimun_val=5, minimun_range=5)
        # if len(peek_ranges_croped) != 1:
        #     _logger.info('Find not only one row content in the file [{}]'.format(img))
        #     return False
        # if int((peek_ranges_croped[0][-1]) * 1.5) < croped_height:
        #     _logger.info('There are not only one row in the file [{}]'.format(img))
        #     return False
        old_im.save(os.path.join(crop_saved_path, file_name))
        return True
    except Exception as e:
        _logger.info('Can not open the file [{}]'.format(img))
        return False


def resize_img(l):
    """ resize the img based the height  
    Args:
        filename: file name with extension
        img_path: input file name directory
        resized_img_path: save directory for the processed image
        height_list: target height 
    """

    filename, img_path, resized_img_path, height_list = l
    try:
        img = Image.open(img_path)
        width, height = img.size
        distance_list = [abs(height - i) for i in height_list]
        target_height = height_list[distance_list.index(min(distance_list))]
        sacle = height / target_height
        target_width = int(width / sacle+1)
        new_size = (target_width, target_height)
        resize_img = img.resize(new_size)
        resize_img.save(os.path.join(resized_img_path, filename))
        return True
    except:
        return False


def generate_image_data(img_path, _logger, downsample=False):
    """ Generate image data and downsample the image data based the "downsample" flage """
    try:
        _image_path_abs = os.path.abspath(img_path)
        img_data = np.asarray(Image.open(_image_path_abs).convert('L'))
        # img=cv2.imread(_image_path_abs)
        # im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # img_data_mean=cv2.adaptiveThreshold(im_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,10)
        if img_data.ndim == 2:
            img_data = img_data[:, :, np.newaxis]
        return img_data
    except:
        pass


def binary_img(img_path):
    try:
        _image_path_abs = os.path.abspath(img_path)
        img = cv2.imread(_image_path_abs)

        # cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        # img_t = cv2.convertScaleAbs(img)
        # fgamma = 120
        # img_gamma = np.power((img_t / 255.0), fgamma) * 255.0
        # im_gray = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2GRAY)

        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_data_mean = cv2.adaptiveThreshold(
            im_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10)
        
        scipy.misc.toimage(img_data_mean).save(_image_path_abs)
        return True
    except:
        return False


def img_aug(img_path):
    """ Image augumentation for the data loader """
    LOW = list(range(20, 70))
    HIEGHT = list(range(70, 100))
    img = cv2.imread(img_path)
    h, w, c = img.shape
    if np.random.rand()> 0.5:
        # resize img
        _img = img.copy()
        resize_scale = random.randint(60, 105) / 100
        if resize_scale > 1.0:
            res = cv2.resize(_img, None, fx=resize_scale, fy=resize_scale,
                             interpolation=cv2.INTER_CUBIC)
            _h, _w, _ = res.shape
            center = (_w/2, _h/2)
            new_w, new_h = (int(center[0] - w / 2),
                            int(center[0] - w / 2) + w), (int(center[1] - h / 2),
                                                          int(center[1] - h / 2) + h)
            img_roi = res[new_h[0]: new_h[1], new_w[0]: new_w[1], :]
        else:
            res = cv2.resize(_img, None, fx=resize_scale, fy=resize_scale,
                             interpolation=cv2.INTER_AREA)
            _h, _w, _ = res.shape
            img_roi = np.ones(shape=(h, w, c), dtype=img.dtype)*255
            img_roi[int(h/2-_h/2):int(h/2-_h/2)+_h, 0:_w, :] = res
        img = img_roi
        del img_roi
    if np.random.rand() > 0.5:
        # rotate the img
        _img = img.copy()
        rotate = random.randint(0, 50)/100
        M = cv2.getRotationMatrix2D((h / 2, w / 2), rotate, 1)
        dst = cv2.warpAffine(
            src=_img, M=M, dsize=(w, h),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        img = dst
        del dst

    if np.random.rand() > 0.5:
        # 平移变换
        _img = img.copy()
        v_d = random.randint(0, 15)
        h_d = random.randint(-int(0.2 * h), int(0.2 * h))
        M = np.float32([[1, 0, v_d], [0, 1, h_d]])
        dst = cv2.warpAffine(
            src=_img, M=M, dsize=(w, h),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        img = dst
        del dst
    if np.random.rand() > 0.5:
        # add background for the input image
        _img = img.copy()
        b, g, r = cv2.split(_img)
        _l = random.choice(LOW)
        _h = random.choice(HIEGHT)
        bot_b = random.randint(int(_l), int(_h), size=(h, w))  # 指定生成随机数范围和生成的多维数组大小
        bot_g = random.randint(int(_l), int(_h), size=(h, w))
        bot_r = random.randint(int(_l), int(_h), size=(h, w))
        bottom = cv2.merge([bot_b, bot_g, bot_r])
        bottom = np.asarray(bottom, dtype=_img.dtype)
        alpha = random.randint(30, 100)/100
        beta = 1-alpha
        img_add = cv2.addWeighted(_img, alpha, bottom, beta, 0)
        img = img_add
        del img_add
    if np.random.rand() > 0.7:
        # 增加椒盐噪声
        _img = img.copy()
        SNR = random.randint(90, 99) / 100
        mask = random.choice((0, 1, 2), size=(h, w, 1), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
        mask = np.repeat(mask, c, axis=-1)     # 按channel 复制到 与img具有相同的shape
        _img[mask == 1] = 255    # 盐噪声
        _img[mask == 2] = 0  # 椒噪声
        img = _img
        del mask, _img
    # NOTE 最后一步在转换为灰度图
    out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return out


def downsample_image(img, output_path, ratio=2):
    """Downsample image by ratio"""
    assert ratio >= 1, ratio
    if ratio == 1:
        return True
    old_im = Image.open(img)
    old_size = old_im.size
    new_size = (int(old_size[0]/ratio), int(old_size[1]/ratio))

    new_im = old_im.resize(new_size, Image.LANCZOS)
    new_im.save(output_path)
    return True


def convert_to_png(formula, dir_output, name, quality=100, density=200,
                   down_ratio=2, buckets=None):
    """Converts LaTeX to png image

    Args:
        formula: (string) of latex
        dir_output: (string) path to output directory
        name: (string) name of file
        down_ratio: (int) downsampling ratio
        buckets: list of tuples (list of sizes) to produce similar shape images

    """
    # write formula into a .tex file
    with open(dir_output + "{}.tex".format(name), "w") as f:
        f.write(
            r"""\documentclass[preview]{standalone}
    \begin{document}
        $$ %s $$
    \end{document}""" % (formula))

    # call pdflatex to create pdf
    run("pdflatex -interaction=nonstopmode -output-directory={} {}".format(
        dir_output, dir_output+"{}.tex".format(name)), TIMEOUT)

    # call magick to convert the pdf into a png file
    run("magick convert -density {} -quality {} {} {}".format(density, quality,
                                                              dir_output+"{}.pdf".format(name), dir_output+"{}.png".format(name)), TIMEOUT)

    # cropping and downsampling
    img_path = dir_output + "{}.png".format(name)
    file_name = "{}.png".format(name)
    try:
        crop_image(img_path, img_path)
        padding_img(img_path, img_path, buckets=buckets)
        downsample_image(img_path, img_path, down_ratio)
        clean(dir_output, name)

        return "{}.png".format(name)

    except Exception as e:
        clean(dir_output, name)
        return False


def clean(dir_output, name):
    delete_file(dir_output+"{}.aux".format(name))
    delete_file(dir_output+"{}.log".format(name))
    delete_file(dir_output+"{}.pdf".format(name))
    delete_file(dir_output+"{}.tex".format(name))


def build_image(item):
    idx, form, dir_images, quality, density, down_ratio, buckets = item
    name = str(idx)
    path_img = convert_to_png(form, dir_images, name, quality, density,
                              down_ratio, buckets)
    return (path_img, idx)


def build_images(formulas, dir_images, quality=100, density=200, down_ratio=2,
                 buckets=None, n_threads=4):
    """Parallel procedure to produce images from formulas

    If some of the images have already been produced, does not recompile them.

    Args:
        formulas: (dict) idx -> string

    Returns:
        list of (path_img, idx). If an exception was raised during the image
            generation, path_img = False
    """
    init_dir(dir_images)
    existing_idx = sorted(set([int(file_name.split('.')[0]) for file_name in
                               get_files(dir_images) if file_name.split('.')[-1] == "png"]))

    pool = Pool(n_threads)
    result = pool.map(build_image, [(idx, form, dir_images, quality, density,
                                     down_ratio, buckets) for idx, form in formulas.items()
                                    if idx not in existing_idx])
    pool.close()
    pool.join()

    result += [(str(idx) + ".png", idx) for idx in existing_idx]

    return result


def image_process(
        input_dir, preprocess_dir, render_out, file_name, target_height, bucket_size, _logger):
    """ 
    process the image and contains forur step:
        if the input file is the format ".png"
        - crop image
        - resize image based target height 
        - padding image based bucket size
        - downsample image 
        elif the input file is the format ".pdf"
        - convert the pdf to png 
        - crop the image 
        - resize image based target height 
        - padding image based bucket size
        - downsample image 
    Args:
        input_dir: input image folder
        preprocess_dir: preprocess folder for the image
        file_name: file name with the extension
    """

    # convert to binary and and adaptive threshold
    flage = binary_img(os.path.join(input_dir, file_name))

    if not flage:
        return False
    params_croped = (
        os.path.join(input_dir, file_name),
        preprocess_dir, file_name, _logger)
    # crop image 
    flage = crop_image(params_croped)
    if not flage:
        _logger.info('Can note crop the image [{:s}]'.format(os.path.join(input_dir, file_name)))
        return False
    # resize the image based target
    params_resize = (file_name, os.path.join(preprocess_dir,
                                             file_name), preprocess_dir, target_height)
    flage = resize_img(params_resize)
    if not flage:
        _logger.info('Can not resize the image [{:s}]'.format(os.path.join(preprocess_dir,
                                                                           file_name)))
        return False
    # pad image based bucket size
    # params_padding = (
    #     file_name, os.path.join(preprocess_dir, file_name),
    #     bucket_size, preprocess_dir)

    # flage = padding_img(params_padding)
    # if not flage:
    #     return False

    # generate image and downsample

    return True
