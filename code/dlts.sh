#!/bin/sh

EXP_ID=$1

NEstimators=$2
MaxDepth=$3
TreeMethod=$4

WDATA_DIR=/data/zhuoyu/tc3d/wdata
OUTPUT_DIR=$WDATA_DIR/$EXP_ID
mkdir $OUTPUT_DIR


sudo apt-get install zip -y
sudo apt-get install -y libgomp1

sudo mkdir /zhuoyu_exp/
sudo chmod 777 /zhuoyu_exp

sudo pip install pip --upgrade
sudo pip install pandas click joblib sklearn
sudo pip install --pre xgboost

EXP_ROOT_DIR=/zhuoyu_exp/work
mkdir $EXP_ROOT_DIR
cd $EXP_ROOT_DIR

#code
cp -r /data/zhuoyu/tc3d/code/tc3d/code $EXP_ROOT_DIR/



#data
cp -r /data/zhuoyu/tc3d/cv_data/0/  $EXP_ROOT_DIR/
mv $EXP_ROOT_DIR/0 $EXP_ROOT_DIR/data

#running
TRAIN_DATA=$EXP_ROOT_DIR/data/train/model
TRAIN_GT=$EXP_ROOT_DIR/data/train/gt
PRED_DATA=$EXP_ROOT_DIR/data/dev/model
PRED_ANS=$EXP_ROOT_DIR/data/dev/gt
PRED_OUT=$OUTPUT_DIR/res
mkdir $PRED_OUT
PRED_OUT_MODEL=$OUTPUT_DIR/model
mkdir $PRED_OUT_MODEL
MODEL_FILE=$PRED_OUT_MODEL/model.bin

cd $EXP_ROOT_DIR/code
python model.py train ${TRAIN_DATA} ${TRAIN_GT} ${MODEL_FILE} ${NEstimators} ${MaxDepth} ${TreeMethod} >> $OUTPUT_DIR/log
python model.py predict-all ${MODEL_FILE} ${PRED_DATA} ${PRED_OUT} >> $OUTPUT_DIR/log
python score.py score-all  ${PRED_OUT} ${PRED_ANS} >> $OUTPUT_DIR/log