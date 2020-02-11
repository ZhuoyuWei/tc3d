#/bin/sh

k=$1

TRAIN_DATA=/data/zhuoyu/3d/cv_data/${k}/train/model
TRAIN_GT=/data/zhuoyu/3d/cv_data/${k}/train/gt
PRED_DATA=/data/zhuoyu/3d/cv_data/${k}/dev/model
PRED_ANS=/data/zhuoyu/3d/cv_data/${k}/dev/gt
PRED_OUT=/data/zhuoyu/3d/output/out/${k}/
mkdir $PRED_OUT
mkdir /data/zhuoyu/3d/output/model/${k}
MODEL_FILE=/data/zhuoyu/3d/output/model/${k}/model.bin

#python model.py train ${TRAIN_DATA} ${TRAIN_GT} ${MODEL_FILE}
python model.py predict-all ${MODEL_FILE} ${PRED_DATA} ${PRED_OUT}
python score.py score-all  ${PRED_OUT} ${PRED_ANS}

