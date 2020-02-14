FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime


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



COPY ./* /code/
RUN chmod +x /code/*
RUN ls -l /code/