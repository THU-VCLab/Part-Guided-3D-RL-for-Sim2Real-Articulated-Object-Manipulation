FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04 AS runtime
ENV DEBIAN_FRONTEND=nonintercative
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
WORKDIR /root

RUN apt-get update
RUN apt-get install -y wget nano htop tmux libsparsehash-dev rsync wget libsm6 libxext6 git rsync sudo ssh vim nano unzip zip pv gcp swig
RUN mkdir -p /usr/share/vulkan/icd.d
RUN echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.1.84"}}' > /usr/share/vulkan/icd.d/nvidia_icd.json

RUN wget \
    https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py38_4.12.0-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py38_4.12.0-Linux-x86_64.sh
    
RUN conda --version
COPY arti_mani.yaml .
RUN echo \
'channels: \n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ \n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ \n\
ssl_verify: true' > ~/.condarc
RUN conda env create -f arti_mani.yaml && conda clean --yes --all
# RUN apt-get install -y gnutls-bin

# RUN git config --global https.sslVerify false && git config --global https.postBuffer 1048576000
## Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "arti_mani", "/bin/bash", "-c"]
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir --upgrade numpy && \
#     pip install --no-cache-dir https://storage1.ucsd.edu/wheels/sapien-dev/sapien-2.0.0.dev20220225-cp38-cp38-manylinux2014_x86_64.whl && \
#     pip install --upgrade git+https://gitee.com/pwxie/torchsparse.git@v1.4.0