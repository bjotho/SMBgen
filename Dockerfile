FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean

RUN ...
