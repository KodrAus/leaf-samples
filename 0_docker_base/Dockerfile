FROM nvidia/cuda:cudnn-devel

ENV USER root
ENV CAPNPROTO_VERSION=0.5.3

RUN cd /var/tmp && apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libblas-dev \
    liblapack-dev \
    libssl-dev && \
  curl -sSf https://static.rust-lang.org/rustup.sh | sh -s -- --without=rust-docs --channel=nightly && \
  curl -sO https://capnproto.org/capnproto-c++-$CAPNPROTO_VERSION.tar.gz && \
  tar zxf capnproto-c++-$CAPNPROTO_VERSION.tar.gz && \
  cd capnproto-c++-$CAPNPROTO_VERSION && \
  ./configure && \
  make -j6 check && \
  sudo make install && \
  DEBIAN_FRONTEND=noninteractive apt-get remove --purge -y curl && \
  DEBIAN_FRONTEND=noninteractive apt-get autoremove -y && \
  rm -rf \
    /var/lib/apt/lists/* \
    /tmp/* \
    /var/tmp/* && \
  mkdir /source

VOLUME ["/source"]
WORKDIR /source
CMD ["/bin/bash"]