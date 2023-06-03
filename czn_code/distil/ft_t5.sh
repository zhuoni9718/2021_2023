#!/bin/bash
#SBATCH -J rt5                              # 作业名为 test
#SBATCH -o ./out/rt5.out                           # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 24:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:a100-pcie-40gb:1
source ~/.bashrc


# 设置运行环境
conda activate q2k

python ft_t5.py
