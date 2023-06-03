import os
import json
import logging
from whoosh.fields import *
from whoosh import index, qparser
from whoosh.qparser import QueryParser
from whoosh.writing import AsyncWriter
from whoosh import scoring
import jieba
import jieba.analyse as ana
import re
class Engine(object):
    """构建本地库的搜索引擎.
    1.建立索引和模式对象
    2.写入索引文件
    3.提供搜索接口
    ------------------------------------
    'logger': logger.
    'index_dir: ', search index directory.
    'docs_size: ', the number of docs contained in local index corpus, -1 for all corpus.
    'qas_size: ', the number of qas contained in local index corpus, -1 for all corpus.
    'rebuild': if rebuild the index directory.
    """

    def __init__(self, logger, index_dir="./", docs_size=30000, qas_size=30000, bs_size=30000, rebuild=False):
        # 获取日志
        self.logger = logger

        self.docs_size = docs_size
        self.qas_size = qas_size
        self.bs_size = bs_size

        # 建立索引和模式对象
        '''
        question=question,\
                                            id=d['id'], \
                                            fact=fact,\
                                            description=write_question,
                                            answer=choice
        '''
        self.schema = Schema(question=TEXT(stored=True), id=ID(stored=True), fact=TEXT(stored=True),choices=TEXT(stored=True),answer =TEXT(stored=True) ,description=TEXT(stored=True))
        self.index_dir = index_dir

        # self.bs_ix = index.create_in(index_dir, self.schema, indexname=self.qc)
        # 写入索引内容
        # self.logger.info("writing bs {} corpus to index...".format(self.qc))
        # self.bs_writer = self.bs_ix.writer()

        # 建立dst_qs库
        if rebuild or not index.exists_in(index_dir, indexname='obqa'):
            self.dst_ix = index.create_in(index_dir, self.schema, indexname='obqa')
            # 写入索引内容
            self.logger.info("writing dst corpus to index...")
            self.dst_writer = self.dst_ix.writer()
            self._write_dst_corpus()
            self.dst_writer.commit()
            self.logger.info("writing dst corpus size is {}".format(self.dst_ix.doc_count()))
            self.logger.info("writing dst corpus done!")
        else:
            self.dst_ix = index.open_dir(index_dir, indexname='obqa')
            self.logger.info("reading dst corpus size is {}".format(self.dst_ix.doc_count()))

    def build_dst_idx(self, index_dir="search_copus"):
        # 建立索引和模式对象
        # schema = Schema(question=TEXT(stored=True), id=ID(stored=True), answer=TEXT(stored=True), question_classification=TEXT(stored=True))

        # 建立索引存储目录
        if not os.path.exists(index_dir):
            os.mkdir(index_dir)

            
        self.dst_ix = index.create_in(index_dir, self.schema, indexname='obqa')
        # 写入索引内容
        # self.logger.info("writing dst corpus to index...)
        self.dst_writer = self.dst_ix.writer()
        self._write_dst_corpus()
        self.dst_writer.commit()
        self.logger.info("writing obqa corpus size is {}".format(self.dst_ix.doc_count()))
        self.logger.info("writing obqa corpus done!")

    def search_dst(self, question, limit=1, limit_score=20.0):
        """根据question检索相关qa pairs.
        "question": str type. question str.
        "limit": int type, returned contexts num.
        "limit_score": float type, match score of every searched qas.
        return
            "res": dict type. It has two keys:
                "question": a list of str, segmented question.
                "qas": a list of list of dicts, several contexts related to the question.
            if return None, it represents no related qas.
        """
        # print(question)
        ##分词、去除停用词
        # seg_question = list(jieba.cut(question))

        # ana.set_stop_words('./QuestionAnswering/deeplearning_models/dst_qs/search/stopword.txt')
        seg_question=ana.extract_tags(question)
        # print(question, self.qc)
        # print("----------seg qus--------:",seg_question)
        # exit()
        try:
            searcher = self.dst_ix.searcher(weighting=scoring.BM25F)
            # 解析query
            # 检索"question"和"answer"的field,指标是BM25
            og = qparser.OrGroup.factory(0.9)
            parser = qparser.QueryParser("description", schema=self.dst_ix.schema, group=og)
            parser = qparser.QueryParser("question", schema=self.dst_ix.schema, group=og)

            #此处的query是分词之后，再用空格拼接起来的
            query = parser.parse(" ".join(seg_question))
            
            # query = " ".join(seg_question)
            # print("-----------query----------:",type(query))  #<class 'whoosh.query.compound.Or'>
            # print("-----------query----------:",query)
            # print("\n")
            result = searcher.search(query, limit=limit) #limit是返回个数限制
            if result.is_empty():
                print('result is empty')
                return None

            # print(result)
            # print("----")
            # 抽取返回结果信息
            res = {}
            res["question"] = question  
            qas = []
            for i, hit in enumerate(result):

                c = {}
                c["question"] = hit["question"]
                c["fact"]=hit["fact"]
                c['choices'] = hit['choices']
                c['answer'] = hit['answer']
                c["match_score"] = result.score(i)
                # if score is not larger than limit_score, filter it
                # print(c)
                qas.append(c)
            if len(qas) == 0:
                return None
            else:
                res["qas"] = qas
                return res
        finally:
            searcher.close()

    def _write_dst_corpus(self, data_dir="./"):
        """写qas库的索引内容"""
        count = 0
        # ana.set_stop_words('./QuestionAnswering/deeplearning_models/dst_qs/search/stopword.txt')

        with open(os.path.join(data_dir, 'q_f_all.jsonl'), 'r', encoding="utf-8") as f:
            for line in f:
                d=json.loads(line)
                # write_query=ana.extract_tags(d["query"])
                #需要过滤问题中的个人信息    答案一般为官方给出，不涉及个人信息
                question=d["question"]['stem']
                fact=d['fact1']
                choices=d['question']['choices']
                choice_str = ''
                answer_idx = ['A','B','C','D'].index(d['answerKey'])
                answer = choices[answer_idx]['text']
                for choice in choices:
                    choice_str = choice_str + '|' + choice['text']
                write_question=ana.extract_tags(question)
                write_question=" ".join(write_question)
                # 需要把答案dict处理成文本才能插入
                # choice_num=4
                # answer_list=[]
                # for i in range(choice_num):
                #     answer_list.append(choice[i]['text'])

                # 此处在索引中存入 问题 fact  答案选项 id   
                # description代表分词、提取完关键词之后的问题 用来搜索
                self.dst_writer.add_document(question=question,\
                                            id=d['id'], \
                                            fact=fact,\
                                            choices = choice_str,\
                                            answer = answer,\
                                            description=write_question,
                                            )

                count += 1
                # if count != -1 and count == self.bs_size:
                # return


if __name__=='__main__':
    logger = logging.getLogger("MC")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # run engine
    engine = Engine(logger)
    #写索引
    # engine.build_dst_idx()
    #检索
    question='You can share files with someone if you have a connection to a what?'
    engine.search_dst(question, limit=3)