#!/bin/bash
#SBATCH -J rankprompt                              # 作业名为 test
#SBATCH -o ./out/04/rankprompt.out                           # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 8:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH -w gpu09                                  # 指定运行作业的节点是 gpu06，若不填写系统自动分配节点
#SBATCH --gres=gpu:tesla_p100-pcie-16gb:1
source ~/.bashrc

# 设置运行环境
conda activate q2k
# conda activate 40g2

# 输入要执行的命令
# python buildprompt.py   #生成prompt
# python  predict.py    
# python test_train.py               
python rank_prompt.py
# python prompt_for_qc.py
# python conceptgen.py