FROM ubuntu:20.04

# Install requirements
RUN apt-get update && apt-get install -y \
    # OpenCV dependencies
    build-essential git curl cmake clang llvm build-essential libgtk2.0-dev pkg-config libavcodec-dev  \
    libavformat-dev  libswscale-dev python3-dev python3-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev  \
    libtiff-dev  libdc1394-22-dev libcanberra-gtk-module libcanberra-gtk3-module

# Clone, build and install OpenCV
RUN git clone https://github.com/opencv/opencv.git && \
    cd /opencv && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j"$(nproc)" && \
    make install

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y && \
    echo 'source $HOME/.cargo/env' >> $HOME/.bashrc

COPY . /home/user/cvldetector
WORKDIR /home/user/cvldetector

CMD cargo build

ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
