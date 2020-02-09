FROM ubuntu:latest

RUN  apt update -y &&\
    apt install -y python3.6 curl wget openssl python3-distutils lsb-release gnupg locales && \
    mkdir pip && cd pip && wget https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py
RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
	&& locale-gen en_US.utf8 \
	&& /usr/sbin/update-locale LANG=en_US.UTF-8
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
RUN pip install pandas click joblib sklearn
RUN cd /usr/bin && ln -s python3.6 python && cd ~
COPY ./* /code/
RUN chmod +x /code/* 