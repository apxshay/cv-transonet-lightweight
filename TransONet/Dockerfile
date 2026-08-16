# Start your image with a node base image
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir /workspace

#here create the folders you need inside the container
RUN mkdir /dataset
RUN mkdir /synthetic_room_dataset
RUN mkdir /watertight

RUN apt-get update && apt-get install -y openssh-server


# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get install ffmpeg -y
RUN apt-get install libsm6 -y
RUN apt-get install libxext6  -y


# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH


ARG root_password
RUN apt-get update || echo "OK" && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo "root:${root_password}" | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

EXPOSE 22

# create non-root user
ARG USERNAME=yourusername
ARG USER_UID=youruserid
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
#USER $USERNAME

ADD environment.yaml /tmp/environment.yml
RUN /bin/bash --login
RUN /opt/conda/bin/conda env create -f /tmp/environment.yml

RUN conda info --envs

SHELL ["conda", "update", "-n", "base", "-c", "defaults", "conda"]
SHELL ["conda", "run", "-n", "neuralblox", "/bin/bash", "-c"]

RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install torch-scatter -f   https://download.pytorch.org/whl/torch-1.12.1+cu113.html #https://data.pyg.org/whl/torch-1.12.1+cu113.html
RUN pip install torchinfo torchsummary torchviz


RUN sudo cp /opt/conda/envs/neuralblox/lib/libstdc++.so.6 /lib/x86_64-linux-gnu


# set permissions for each folder created before

RUN sudo chattr -i /dataset
RUN lsattr /dataset
RUN sudo chattr -i /synthetic_room_dataset
RUN lsattr /synthetic_room_dataset
RUN sudo chattr -i /watertight
RUN lsattr /watertight


#delete sudo
RUN sudo chown -R $USER_UID:$USER_GID /workspace

#repeat for each folder created before
RUN sudo chown -R $USER_UID:$USER_GID /dataset
RUN sudo chown -R $USER_UID:$USER_GID /synthetic_room_dataset
RUN sudo chown -R $USER_UID:$USER_GID /watertight

WORKDIR /workspace


#commands on terminal to use everytime you start the docker container:
"""
 #create the base env of conda
 eval "$(conda shell.bash hook)"
 #activate your env
 conda activate neuralblox
"""





# ********************************************************
# python train.py configs/pointcloud/shapenet/shapenet_dynamic_3plane_final.yaml
# ********************************************************
#tensorboard
#check what PID is running on port 6006-> lsof -i:6006
#kill <PID>
#pycharm ->  tensorboard --bind_all --logdir out/pointcloud/shapenet_dynamic_3plane_final/logs
#locaalhost -> http://192.168.1.74:8083/
# ********************************************************
