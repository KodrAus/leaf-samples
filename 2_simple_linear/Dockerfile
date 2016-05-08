FROM kodraus/rust-leaf:latest

WORKDIR /
COPY leaf-demo-2/ source/
RUN cd /source && cargo build --release &&\
  cp -r target/release /output
CMD /output/leaf-demo-2