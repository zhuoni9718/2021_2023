#!/bin/bash
#SBATCH -J  t5_ra_test                              # 作业名为 test
#SBATCH -o ./out/05/gen_ra_t5_wotrain.out                           # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 12:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:a100-pcie-40gb:1
source ~/.bashrc
echo "Current time: $(date)"
# 设置运行环境
conda activate q2k

# model_name=facebook/bart-large
# model_name=gpt2-large
model_name=t5-large
dataset_class=RA
epoch=10
batch_size=4
learning_rate=1e-5
best_epoch=0
python gen_model_frame.py \
    --model_name $model_name \
    --dataset_class $dataset_class\
    --train_data ./input/csqa/right.train.gpt.jsonl \
    --dev_data /users5/znchen/distil/input/csqa/right.valid.gpt.jsonl \
    --test_data ./input/csqa/test.gpt.jsonl\
    --epochs $epoch \
    --best_epoch $best_epoch\
    --test\
    --batch_size $batch_size\
    --learning_rate $learning_rate\
    --seed 42\
    # --train


    # --test_data ./input/csqa/test.gpt.jsonl \

# warmup 500

# Dataloader_classes = {
#     'RA':DistlCSQADatasetRA,
#     'WOCOT':DistlCSQADataset,
#     'R':DistlCSQADatasetR,
#     'WOCOT-base':DistlCSQADataset
# }