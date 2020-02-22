#!/bin/sh

EXP_ID=$1

NEstimators=$2
MaxDepth=$3
TreeMethod=$4
n_job=$5
sample_rate=$6


WDATA_DIR=/data/zhuoyu/tc3d/wdata
OUTPUT_DIR=$WDATA_DIR/$EXP_ID
mkdir $OUTPUT_DIR


export LC_ALL=C.UTF-8
export LANG=C.UTF-8

sudo apt-get install zip -y
sudo apt-get install -y libgomp1

sudo mkdir /zhuoyu_exp/
sudo chmod 777 /zhuoyu_exp

sudo pip install pip --upgrade
sudo pip install pandas click joblib sklearn
#sudo pip install --pre xgboost
sudo pip install --pre https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/xgboost-1.0.0_SNAPSHOT%2Bda6e74f7bb7727ae60b37b1dc6f86ef98d350887-py2.py3-none-manylinux1_x86_64.whl

EXP_ROOT_DIR=/zhuoyu_exp/work
mkdir $EXP_ROOT_DIR
cd $EXP_ROOT_DIR

#code
cp -r /data/zhuoyu/tc3d/code/tc3d/code $EXP_ROOT_DIR/



#data
mkdir  $EXP_ROOT_DIR/data
cp -r /data/zhuoyu/tc3d/model  $EXP_ROOT_DIR/data/
cp -r /data/zhuoyu/tc3d/gt  $EXP_ROOT_DIR/data/


#running
TRAIN_DATA=$EXP_ROOT_DIR/data/model
TRAIN_GT=$EXP_ROOT_DIR/data/gt
PRED_DATA=$EXP_ROOT_DIR/data/dev/model
PRED_ANS=$EXP_ROOT_DIR/data/dev/gt
PRED_OUT=$OUTPUT_DIR/res
mkdir $PRED_OUT
PRED_OUT_MODEL=$OUTPUT_DIR/model
mkdir $PRED_OUT_MODEL
MODEL_FILE=$PRED_OUT_MODEL/model.bin

cd $EXP_ROOT_DIR/code
python model.py train ${TRAIN_DATA} ${TRAIN_GT} ${MODEL_FILE} ${NEstimators} ${MaxDepth} ${TreeMethod} ${n_job} ${sample_rate} >> $OUTPUT_DIR/log
