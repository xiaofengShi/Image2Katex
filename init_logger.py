'''
Filename: init_logger.py
Project: image2katex
File Created: Sunday, 9th December 2018 6:11:11 pm
Author: xiaofeng (sxf1052566766@163.com)
--------------------------
Last Modified: Sunday, 9th December 2018 6:11:15 pm
Modified By: xiaofeng (sxf1052566766@163.com)
---------------------------
'''

import logging
import os


def get_logger(_loggerDir, log_path, logger_name):
    _LogFile = os.path.join(_loggerDir, log_path)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    # ccreate file handler which logs even debug messages
    fh = logging.FileHandler(_LogFile)
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    _LogFormat = logging.Formatter("%(asctime)2s -%(name)-12s:  %(levelname)-10s - %(message)s")

    fh.setFormatter(_LogFormat)
    console.setFormatter(_LogFormat)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(console)
    return logger
