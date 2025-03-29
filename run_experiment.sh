#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0  # 如果有GPU，使用第一个GPU

# 创建必要的目录
mkdir -p mappo_signaling_checkpoints
mkdir -p evaluation_results

# 运行训练
echo "开始训练..."
python train.py

# 等待训练完成
echo "训练完成，开始评估..."

# 运行评估
echo "开始评估..."
python evaluate.py

echo "实验完成！" 