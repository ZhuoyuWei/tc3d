#!/bin/bash

TRAIN_DATA='/data/input/train'
TRAIN_GT='/data/gt/train'
PRED_DATA='/data/input/pred'
PRED_OUT='/data/output/pred'
MODEL_FILE='/tmp/model.bin'

#pip install xgboost

python /code/model.py train ${TRAIN_DATA} ${TRAIN_GT} ${MODEL_FILE}