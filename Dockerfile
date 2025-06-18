FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y build-essential git curl cmake clang libclang-dev \
    llvm llvm-dev python3-dev python3-numpy libgtk2.0-dev pkg-config libavcodec-dev \
    libavformat-dev libswscale-dev libtbb2 libtbb-dev libcanberra-gtk-module \
    libcanberra-gtk3-module libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev libopencv-dev

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    ln -s $HOME/.cargo/bin/* /usr/bin/

RUN rustup default nightly-unknown-linux-gnu

COPY . /home/user/cvlcore
WORKDIR /home/user/cvlcore

RUN cargo build --verbose --release --all-targets --jobs $(nproc)
RUN cargo doc --verbose && mv target/doc target/release/
RUN tar -zcvf cvlcore.tar.gz target/release/*

ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
