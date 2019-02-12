FROM python:3.6.3-jessie

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PATH /usr/local/nvidia/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Tell nvidia-docker the driver spec that we need as well as to
# use all available devices, which are mounted at /usr/local/nvidia.
# The LABEL supports an older version of nvidia-docker, the env
# variables a newer one.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

WORKDIR /rc-experiments/

# Use this line here because currently the downloading the 1.0.0 version in docker always crash.
# The reason might be that PyTorch 1.0 uses different installation file for different CPU/GPU settings.
# When building image locally on mac, there is no GPU support.
# And Deehru found that the performance becomes very unstable on torch 1.0. This needs more tests to confirm.
# So we just use the old version of Pytorch here.
RUN pip install torch==0.4.1

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN spacy download en_core_web_sm
COPY reading_comprehension reading_comprehension

COPY env_vars env_vars