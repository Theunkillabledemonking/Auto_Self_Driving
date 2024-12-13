FROM nvcr.io/nvidia/l4t-base:r32.7.1

ARG uid=1000
ARG gid=1000
ARG gid_gpio=999

RUN groupadd -g $gid_gpio gpio && \
    groupadd -g $gid user && \
    useradd -m -u $uid -g $gid user && \
    usermod -aG gpio user

RUN apt update && apt install -y python3 python3-pip && \
    python3-dev python3-pip python3-setuptools \
    && pip3 install Cython \
    && git clone https://github.com/NVIDIA/jetson-gpio.git /tmp/jetson-gpio \
    && cd /tmp/jetson-gpio && python3 setup.py install \
    && rm -rf /tmp/jetson-gpio

