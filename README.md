# Dual-branch Hybrid Learning Network for Unbiased Scene Graph Generation
This repository contains code for the paper ['Dual-branch Hybrid Learning Network for Unbiased Scene Graph Generation'](https://arxiv.org/abs/2207.07913).

## Installation
Check [INSTALL.md](./INSTALL.md) for installation instructions.
## Dataset
Check [DATASET.md](./DATASET.md) for instructions of dataset preprocessing.

## Device
All our experiments are conducted on one NVIDIA GeForce RTX 3090, if you wanna run it on your own device, make sure to follow distributed training instructions in [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).
## Train
We provide scripts for training the models
```
#!/usr/bin/env bash
export PYTHONPATH=/home/zhengchaofan/lib/apex:/home/zhengchaofan/lib/cocoapi:/home/zhengchaofan/code/Scene-Graph-Benchmark.pytorch-master=:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=3
export NUM_GUP=1
echo "TRAINING!!!!"

MODEL_NAME='TransformerPredictor_test1'
mkdir ./checkpoints/${MODEL_NAME}/
cp ./tools/relation_train_net.py ./checkpoints/${MODEL_NAME}/
cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py ./checkpoints/${MODEL_NAME}/
cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/model_transformer.py ./checkpoints/${MODEL_NAME}/
cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/loss.py ./checkpoints/${MODEL_NAME}/
cp ./scripts/train.sh ./checkpoints/${MODEL_NAME}/
cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/relation_head.py ./checkpoints/${MODEL_NAME}

python \
tools/relation_train_net.py \
--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
DTYPE "float32" \
SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH $NUM_GUP \
SOLVER.MAX_ITER 40000 SOLVER.BASE_LR 1e-3 \
SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
SOLVER.STEPS "(20000, 26000, 36000, )" SOLVER.VAL_PERIOD 4000 \
SOLVER.CHECKPOINT_PERIOD 10000 GLOVE_DIR ./datasets/vg/ \
MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
SOLVER.PRE_VAL False \
```

## Test
We provide scripts for testing the models
```
#!/usr/bin/env bash
# export PYTHONPATH=/home/chaofan/lib/apex:/home/chaofan/lib/cocoapi:/home/chaofan/myCode/Scene-Graph-Benchmark.pytorch-master=:$PYTHONPATH
export PYTHONPATH=/home/zhengchaofan/lib/apex:/home/zhengchaofan/lib/cocoapi:/home/zhengchaofan/code/Scene-Graph-Benchmark.pytorch-master=:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=3
export NUM_GUP=1
echo "Testing!!!!"
MODEL_NAME="TransformerPredictor_test1"
python \
        tools/relation_test_net.py \
        --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
        MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
        TEST.IMS_PER_BATCH 1 DTYPE "float32" \
        GLOVE_DIR ./datasets/vg/ \
        MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
        MODEL.WEIGHT ./checkpoints/${MODEL_NAME}/model_final.pth \
        OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
        TEST.ALLOW_LOAD_FROM_CACHE False \
```

## Help
Be free to contact me (`zheng_chaofan@foxmail.com`) if you have any questions!

## Acknowledgement
The code is implemented based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).
