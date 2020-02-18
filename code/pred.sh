#!/bin/bash

TRAIN_DATA='/data/input/train'
TRAIN_GT='/data/gt/train'
PRED_DATA='/data/input/pred'
PRED_OUT='/data/output/pred'
MODEL_FILE='/tmp/model.bin'

#pip install xgboost

python /code/model.py predict-all ${MODEL_FILE} ${PRED_DATA} ${PRED_OUT} 0