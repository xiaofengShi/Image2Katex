import logging
import os


def init_logger(_loggerDir=os.path.dirname(__file__),
                log_path='sequence_dataset.log', logger_name='ErrorCheck'):

    _LogFile = os.path.join(_loggerDir, log_path)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    # ccreate file handler which logs even debug messages
    fh = logging.FileHandler(_LogFile)
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    # the logger level info to display on the command
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
