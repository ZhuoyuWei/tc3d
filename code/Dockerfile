FROM nvidia/cuda:10.0-base-ubuntu16.04



ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN apt-get install -y libgomp1

RUN pip install pip --upgrade
RUN pip install pandas click joblib sklearn
RUN pip install numpy==1.13.1
RUN pip install scipy
RUN pip install -U scikit-learn
RUN pip install seaborn
RUN pip install matplotlib
RUN pip install --pre xgboost



RUN cd /usr/bin && ln -s python3.6 python && cd ~
COPY ./* /code/
RUN chmod +x /code/* 