# Specify the parent image from which we build
# Starting from https://hub.docker.com/r/pytorch/pytorch/tags
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

LABEL pipeline="Test Pipeline Pytorch-GPU"
LABEL maintainer="Israel Tinini <israel.tininialvarez@sony.com>"
LABEL description="PyTorch2.2 Container with GPU support"

WORKDIR /home

ENV DEBIAN_FRONTEND noninteractive
ENV HDF5_USE_FILE_LOCKING=FALSE

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 git -y
RUN apt-get install python3-pip -y
# RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip install matplotlib>=3 \
                numpy>=1.22\
                scipy>=1.8 \
                tensorboard \
                opencv-python>=4.5

RUN apt-get install wget

