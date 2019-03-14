#!/bin/sh
# setup env for the gpu on the remote server
# source ~/tf-gpu/bin/activate
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


# Get the current dir
# SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)

# Switch to current folder
# cd ${SHELL_FOLDER}
# Run the model
# python3 app.py --device gpu  --crnn Densenet 
