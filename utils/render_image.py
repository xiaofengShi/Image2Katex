# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
#                  Life is short, You need Python!
# [File]         : render_image.py
# [Project]      : shixiaofeng
# [CreatedDate]  : Thursday, 2019-03-14 11:42:56
# [Author]       : shixiaofeng
# ==================================================================
# [LastModified] : Saturday, 2020-04-11 14:38:34
# [ModifiedBy]   : shixiaofeng
# [Descriptions] : 
# 
# ==================================================================
# [ChangeLog]:
# [Date]    	[Author]	[Comments]
# ------------------------------------------------------------------

import glob
import os
import sys
from subprocess import call

DEVNULL = open(os.devnull, "w")
# latex strcuture of the input

RENDER_FORMAT = r'''
\documentclass[12pt]{article}
\pagestyle{empty}
\usepackage{amsmath}
\begin{document}

\begin{displaymath}
%s
\end{displaymath}

\end{document}
'''

RENDERING_SETUPS = [RENDER_FORMAT, "magick convert -density %d -quality %d %s %s"]


def remove_temp_files(name):
    """ Removes .aux, .log, .pdf and .tex files for name """
    
    os.remove(name + ".aux")
    os.remove(name + ".log")
    os.remove(name + ".pdf")
    os.remove(name + ".tex")
    


def latex_to_image(formula, file_name_no_ext, logger, quality=200, density=200):
    """ Turns given formula into images based on RENDERING_SETUPS

    render the image based latex 
    """
    try:
        rend_setup = RENDERING_SETUPS
        full_path = file_name_no_ext
        # Create latex source
        _latex = rend_setup[0] % formula
        # Write latex source
        with open(full_path + ".tex", "w") as f:
            f.write(_latex)

        # Call pdflatex to turn .tex into .pdf
        code = call(["pdflatex", '-interaction=nonstopmode', '-halt-on-error',
                    full_path + ".tex"], stdout=DEVNULL, stderr=DEVNULL)

        if code != 0:
            os.system("rm -rf " + full_path + "*")
            logger.info('The latex is [{:s}] and can not be rendered'.format(formula))
            return False

        # Turn .pdf to .png
        # Handles variable number of places to insert path.
        # i.e. "%s.tex" vs "%s.pdf %s.png"

        cmd_str = (quality, density, full_path + ".pdf", full_path + ".png")
        code = call((rend_setup[1] % cmd_str).split(" "), stdout=DEVNULL, stderr=DEVNULL)
        # Remove files
        try:
            remove_temp_files(full_path)
        except Exception as e:
            # try-except in case one of the previous scripts removes these files
            # already
            logger.info('Can not remove the temp rendering files')
            return False

        # Detect of convert created multiple images -> multi-page PDF

        if code != 0:
            # Error during rendering, remove files and return None
            os.system("rm -rf " + full_path + "*")
            logger.info('Can not convert pdf to png, please check the cmd message: {}'.format(
                (rend_setup[1] % cmd_str).split(" ")))
            return False
        else:
            return True
    except Exception as e:
        print("render image error {}".format(e))
        return False
