FROM kodraus/rust-leaf:latest

WORKDIR /
COPY leaf-demo-1/ source/
RUN cd /source && cargo build --release &&\
  cp -r target/release /output
CMD /output/leaf-demo-1