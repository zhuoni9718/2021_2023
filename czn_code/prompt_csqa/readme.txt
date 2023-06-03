
faq.py用来从obqa知识库中抽取相似问题 存入
buildprompt.py用来将抽取出来的相似问题构造成prompt的形式
gen_k.py 使用构造出的prompt，用预训练模型生成K   
preiction.py利用生成的K 预测答案，计算准确率
