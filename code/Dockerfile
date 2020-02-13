FROM datmo/python-base:py35-cpu

RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
	&& locale-gen en_US.utf8 \
	&& /usr/sbin/update-locale LANG=en_US.UTF-8
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN pip install pandas click joblib sklearn
RUN cd ~
COPY ./* /code/
RUN chmod +x /code/* 