#!/bin/bash
#SBATCH -J cggpt_QC                              # 作业名为 test
#SBATCH -o ./out/05/roberta_ft_pre_gpt_cg.out                           # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 12:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:a100-pcie-40gb:1
source ~/.bashrc

echo "Current time: $(date)"

# 设置运行环境
conda activate q2k

model_name=roberta-large
dataset_class=WOCOT
epoch=8
best_epoch=4
batch_size=8
learning_rate=1e-5
model_dir=./tmp/predict/promptk_QCK/

python predict_frame.py \
    --model_name $model_name \
    --dataset_class $dataset_class\
    --train_data /users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/cg/k_ft_gpt_cg.train.jsonl/ \
    --dev_data users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/cg/k_ft_gpt_cg.jsonl \
    --test_data users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/cg/k_ft_gpt_cg.jsonl \
    --epochs $epoch \
    --test\
    --batch_size $batch_size\
    --learning_rate $learning_rate\
    --best_epoch $best_epoch\
    --model_path $model_dir\
    --dataloader McQ_promptKc\
    --train\
    --seed 42
    # --dataloader McQ_promptKc
    # --test_data /users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/xlnet/k_what_xlnet.jsonl\
    # --test_data /users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/t5/k_Sk_T5.jsonl\
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