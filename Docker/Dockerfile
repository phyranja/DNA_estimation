#File copyed and adapted from HistoBlur (https://github.com/choosehappy/HistoBlur)
#original file by Petros Liakopoulos

#10.2 is the only version of cuda to support both torch 1.6 and cupy
FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04
#FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
#FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

LABEL software="DNA_estim"

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="Etc/UTC"
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    apt-get install -y libgl1 && \
    apt-get install -y build-essential && \
    apt-get install libxrender1 && \
    apt-get install -y libgtk2.0-dev && \
    apt-get install -y openslide-tools && \
    apt-get install -y libjpeg-dev zlib1g-dev &&\
    rm -rf /var/lib/apt/lists/*

    
    
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH
#ENV LD_LIBRARY_PATH /usr/local/cuda-11.8.0/lib64:/usr/local/cuda-11.8.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
#ENV LD_LIBRARY_PATH /usr/local/cuda-9.0/lib64:/usr/local/cuda-9.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-10.2/lib64:/usr/local/cuda-10.2/extras/CUPTI/lib64:$LD_LIBRARY_PATH


#RUN conda env create -f environment.yml
#RUN conda activate hovernet
#RUN apt-get install openslide-tools
RUN pip install --upgrade pip
RUN pip install protobuf==3.19.6
RUN pip install pillow==8.4
RUN pip install opencv-python
RUN pip install shapely
RUN pip install geojson
RUN pip install paquo
RUN paquo get_qupath --install-path /opt 0.3.2

ARG x=1

RUN git clone --recurse-submodules https://github.com/phyranja/DNA_estimation.git
WORKDIR /DNA_estimation
RUN ls hover_net

RUN pip install -r hover_net/requirements.txt
RUN pip install openslide-python==1.1.2
RUN pip install torch==1.6.0 torchvision==0.7.0
RUN pip install numpy==1.23.1
RUN pip install cupy-cuda102


ARG y=2

COPY run_estim.sh .
CMD ./run_estim.sh
