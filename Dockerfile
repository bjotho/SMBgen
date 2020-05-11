FROM ubuntu:18.04
MAINTAINER Bjørn Inge


RUN apt-get update && apt-get install python3 python3-pip git -y
RUN git clone https://github.com/bjotho/SMBgen.git
RUN cd SMBgen && git pull && git checkout dqn_gen
RUN pip3 install -r SMBgen/requirements.txt
RUN apt-get install firefox -y


ENTRYPOINT ["/usr/bin/python3", "SMBgen/main.py"]
