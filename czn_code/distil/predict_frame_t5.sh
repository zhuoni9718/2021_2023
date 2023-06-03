#!/bin/bash
#SBATCH -J t5                              # 作业名为 test
#SBATCH -o ./out/0511/roberta_draw_t5.out                           # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 24:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:a100-pcie-40gb:1
source ~/.bashrc
echo "Current time: $(date)"

# 设置运行环境
conda activate q2k
# 需要改 训练、验证数据集路径 模型保存路径 out文件路径 是否训练参数
model_name=roberta-large
dataset_clas=WOCOT
epoch=10
batch_size=16
learning_rate=5e-6
best_epoch=1
python predict_frame.py \
    --model_name $model_name \
    --dataset_class $dataset_clas\
    --train_data /users5/znchen/distil/res/t5large/r8_train.res.r0.jsonl\
    --test_data /users5/znchen/distil/res/t5large/R8test.res.r0.jsonl\
    --dev_data /users5/znchen/distil/res/t5large/R8test.res.r0.jsonl \
    --epochs $epoch \
    --test\
    --batch_size $batch_size\
    --best_epoch $best_epoch\
    --learning_rate $learning_rate\
    --model_path ./tmp/predict/promptk_QCK_rationale/t5rationaletrain\
    --dataloader McQKC\
    --seed 42\
    --train
    # --test_data /users5/znchen/distil/input/csqa/right.test.gpt.jsonl \
    # --Dataloader 
    # --train_data ./input/csqa/right.train.gpt.jsonl \

    # --test_data /users5/znchen/distil/input/csqa/test.gpt.jsonl \


# MODEL_CLASSES = {
#     'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
#     'gpt2-medium': (GPT2LMHeadModel, GPT2Tokenizer),
#     'gpt2-large': (GPT2LMHeadModel, GPT2Tokenizer),
#     'gpt2-xl': (GPT2LMHeadModel, GPT2Tokenizer),
#     'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
#     't5-large': (T5ForConditionalGeneration, AutoTokenizer),
#     'facebook/bart-large': (BartForConditionalGeneration, BartTokenizer),
# }
# Dataset_classes = {
#     # 'RA':DistlCSQADatasetRA,
#     # 'WOCOT':DistlCSQADataset,
#     # 'R':DistlCSQADatasetR,
#     # 'WOCOT-base':DistlCSQADataset,
#     'McQKC':MultipleChoiceDataset,
#     'McQ_promptKc':MultipleChoiceDatasetForPromptK,
#     'QC':MultipleChoiceDatasetQC
# }