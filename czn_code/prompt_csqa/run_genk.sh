#!/bin/bash
#SBATCH -J t5                              # 作业名为 test
#SBATCH -o ./out/05/gen_10_t5.out                           # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 24:00:00                            # 任务运行的最长时间为 1 小时  
#SBATCH --gres=gpu:tesla_v100s-pcie-32gb:1
source ~/.bashrc
echo "Current time: $(date)"

# 设置运行环境
conda activate q2k
# conda activate 40g2

# 输入要执行的命令
# python buildprompt.py   #生成prompt
# python  predict.py    
# python test_train.py               
python gen_k.py 
# python rank_prompt.py
# python buildprompt.py