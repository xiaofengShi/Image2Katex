import os
import numpy as np
import time
import logging
import sys
import subprocess
import shlex
from shutil import copyfile
import json
from threading import Timer
from os import listdir
from os.path import isfile, join

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.pdf']
DEVNULL = open(os.devnull, "w")


def allow_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_img_list(img_path):

    return [img_dir for img_dir in os.listdir(img_path) if allow_image_file(img_dir)]


def run(cmd, timeout_sec):
    """Run cmd in the shell with timeout"""
    proc = subprocess.Popen(cmd, shell=True)

    def kill_proc(p): return p.kill()
    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        stdout, stderr = proc.communicate()
    finally:
        timer.cancel()


def run_call(cmd, timeout_sec):
    """Run cmd in the shell with timeout"""
    assert isinstance(cmd, list), 'CMD must be format list '
    proc = subprocess.call(cmd, stdout=DEVNULL, stderr=DEVNULL, timeout=timeout_sec)
    return proc


def get_logger(filename):
    """Return instance of logger"""
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


def init_dir(dir_name):
    """Creates directory if it does not exists"""
    if dir_name is not None:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


def init_file(path_file, mode="a"):
    """Makes sure that a given file exists"""
    with open(path_file, mode) as f:
        pass


def get_files(dir_name):
    files = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
    return files


def delete_file(path_file):
    try:
        os.remove(path_file)
    except Exception:
        pass
