#%%
# import jieba.analyse as ana
import jsonlines
from tqdm import tqdm

def getquestion(data_dir):
    questions = []
    # print("111")
    with jsonlines.open(data_dir,'r') as f:
        # print("222")
        for line in f:
            # print("q:",line["question"]["stem"])
            questions.append(line["question"]["stem"])
    # print(len(questions))
    return questions

def get_prompt(dir):
    data_dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/dev.statement.jsonl'
    # print("begin get question")
    questions = getquestion(data_dir)
    # print(len(questions))
    res = []
    idx = 0

    # with jsonlines.open(dir,'r') as f:
    with open(dir,'r') as f:
        for line in tqdm(f):
            # print(int(idx/5))
            question = questions[int(idx/5)]
            idx += 1
            # qc = line['qc']
            # ac = line['ac']
            # entities = " ".join(qc)+" " +" ".join(ac)

            key_words = line
            # print(key_words)
            # print(question)
            # if idx == 10:
                # exit()
            # prompt = 'Generate some knowledge about the entities in the input. Examples: \
            # Input: google gps highway map maps replace replaced service services street atlas \
            # Knowledge: Electronic maps are the modern version of paper atlas.\
            # Input:  city forest fox look looking walk walked flower flowers pretty pretty_flower\
            # Knowledge: Natural habitats are usually away from cities. \
            # Input: connection file files share share_files freeway\
            # Knowledge: Files can be shared over the Internet.\
            # Input:{}\
            # Knowledge:'.format(entities)

            prompt = "Generate some sentence for the concepts in the input. Examples:\n\
Question: Google Maps and other highway and street GPS services have replaced what?\n\
Input: replace service street atlas highway\n\
Knowledge: Electronic maps are the modern version of paper atlas.\n\
Question: The fox walked from the city into the forest, what was it looking for?\n\
Input: habitat forest walk city natural fox\n\
Knowledge: Natural habitats are usually away from cities.\n\
Question: You can share files with someone if you have a connection to a what?\n\
Input: connection computer file share network\n\
Knowledge: Files can be shared over the Internet.\n\
Question: Too many people want exotic snakes. The demand is driving what to carry them?\n\
Input: shops drive snake carry demand pet\n\
Knowledge: Some people raise snakes as pets.\n\
Question: The body guard was good at his duties, he made the person who hired him what?\n\
Input: duty person guard hire job better body\n\
Knowledge: The job of body guards is to ensure the safety and security of the employer.\n\
Question: {}\n\
Input: {}\
Knowledge:".format(key_words,question)


            res.append(prompt)
    with jsonlines.open('entities.txt','w') as f:
        for item in tqdm(res):
            f.write([item])

if __name__ == '__main__':
    dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/qac/csqa.dev.qac.src' #qac
    get_prompt(dir)



    # s = "Google Maps and other highway and street GPS services have replaced what?"
    # s = 'Google Maps and other highway and street GPS services have replaced united states'
    # key_words = ana.extract_tags(s)
# %%
