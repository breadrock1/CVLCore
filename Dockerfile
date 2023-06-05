FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y build-essential git curl cmake clang libclang-dev \
    llvm llvm-dev python3-dev python3-numpy libgtk2.0-dev pkg-config libavcodec-dev \
    libavformat-dev libswscale-dev libtbb2 libtbb-dev libcanberra-gtk-module \
    libcanberra-gtk3-module libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev

RUN git clone https://github.com/opencv/opencv.git && \
    cd /opencv && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j"$(nproc)" && \
    make install

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- --default-toolchain 1.68.2 -y && \
    ln -s $HOME/.cargo/bin/* /usr/bin/

COPY . /home/user/cvlcore
WORKDIR /home/user/cvlcore

RUN cargo build --verbose --release --all-targets --jobs $(nproc)
RUN cargo doc --verbose && mv target/doc target/release/
RUN tar -zcvf cvlcore.tar.gz target/release/*

ENTRYPOINT [ "/bin/bash", "-l", "-c" ]