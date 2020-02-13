#!/bin/sh

EXP_ID=$1

WDATA_DIR=/data/zhuoyu/tc3d/wdata
OUTPUT_DIR=$WDATA_DIR/$EXP_ID
mkdir $OUTPUT_DIR

sudo apt-get install zip -y

sudo mkdir /zhuoyu_exp/
sudo chmod 777 /zhuoyu_exp

EXP_ROOT_DIR=/zhuoyu_exp/work
mkdir $EXP_ROOT_DIR
cd $EXP_ROOT_DIR

#code
cp -r /data/zhuoyu/tc3d/code/tc3d/code $EXP_ROOT_DIR/



#data
