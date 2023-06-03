import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
# from nltk.translate.rouge_score import rouge_n, rouge_l, rouge_scorer
from pycocotools.coco import COCO

# 句子列表
sentences = [
    "The cat sat on the mat.",
    "The dog ate my homework.",
    "It was the best of times, it was the worst of times.",
    "To be or not to be, that is the question.",
    "In the beginning God created the heavens and the earth."
]

# 参考列表，每个句子有多个参考答案
references = [
    ["The cat sat on the mat."],
    ["The dog ate my homework."],
    ["It was the best of times, it was the worst of times."],
    ["To be or not to be, that is the question."],
    ["In the beginning God created the heavens and the earth."]
]

# 测试句子
test_sentence = "The cat ate the homework."


# 加载数据
coco = COCO('annotations/captions_train2014.json')
ann_ids = coco.getAnnIds()
annotations = coco.loadAnns(ann_ids)

# 对句子进行分词
references = [nltk.word_tokenize(annotation['caption'].lower()) for annotation in annotations]
hypotheses = [nltk.word_tokenize('The cat ate the homework.')]  # 假设句子

# 计算ROUGE-2/L、BLEU-3/4、METEOR、CIDEr、SPICE
rouge_scorer = rouge_scorer.RougeScorer(['rouge2', 'rougeL'], use_stemmer=True)
# rouge2_score = rouge_n(hypotheses, references, n=2, scorer=rouge_scorer)['rouge2'].fmeasure
# rougeL_score = rouge_l(hypotheses, references, scorer=rouge_scorer)['rougeL'].fmeasure
bleu3_score = corpus_bleu(references, hypotheses, weights=(0, 0, 1))
bleu4_score = corpus_bleu(references, hypotheses, weights=(0, 0, 0, 1))
meteor_score = meteor_score(references, 'The cat ate the homework.')
