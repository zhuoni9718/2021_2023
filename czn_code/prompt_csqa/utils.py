from http.client import LENGTH_REQUIRED
from multiprocessing import context
import jsonlines

def readCommongen(data_dir):
    #{"concept_set": "field#look#stand", "scene": ["The player stood in the field looking at the batter.", "The coach stands along the field, looking at the goalkeeper.", "I stood and looked across the field, peacefully.", "Someone stands, looking around the empty field."], "reason": ["A baseball player playing in the field is waiting for the pitcher to pitch the ball to the batter. The player will stand ready while watching the batter.", "Soccer is played on a field. Soccer has coaches and goalkeepers.", "You have a better view to look at things when you stand. A field is a setting in which one can be.", "n/a"]}
    # data {question:  choices:  answerKey: groundTruth:}
    data = []
    with jsonlines.open(data_dir,'r') as f:
        for line in f:
            concepts = ' '.join(line["concept_set"].split('#')) 
            data.append({'question':concepts,'choices':'',\
                'answerKey':line["scene"],'gt':line["reason"]})
    return data

def readObqa(data_dir):
    data = []
    # "id": "8-376", "question": {"stem": "Frilled sharks and angler fish live far beneath the surface of the ocean, which is why they are known as", "choices": [{"text": "Deep sea animals", "label": "A"}, {"text": "fish", "label": "B"}, {"text": "Long Sea Fish", "label": "C"}, {"text": "Far Sea Animals", "label": "D"}]}, "fact1": "deep sea animals live deep in the ocean", "humanScore": "0.80", "clarity": "1.80", "turkIdAnonymized": "bdb4135787", "answerKey": "A"}
    # data: {question:  choices:  answerKey: groundTruth:}
    with jsonlines.open(data_dir,'r') as f:
        for line in f:
            data.append({'question':line['question']['stem'],'choices':line["question"]['choices'],\
                'answerKey':line["answerKey"],'gt':line['fact1']})
            
    return data

def readCsqa(data_dir):
    data = []
    with jsonlines.open(data_dir,'r') as f:
        for line in f:
            # print('line: ',line)
            # print({'question':line['question']['stem'],'choices':line['question']['choices'],\
                # 'answerKey':line["answerKey"],'gt':''})
            # exit()
            data.append({'question':line['question']['stem'],'choices':line['question']['choices'],\
                'answerKey':line["answerKey"],'gt':''})
    # print(data)
    return data

def read_k(k_dir):
    k = []
    with jsonlines.open(k_dir,'r') as f:
        for line in f:
            k.append(line)
    return k    


# def init_model(modelname,device):

def concarate(data,k,k_num,mode = 'bert'):
    labels,contexts  =  [] , []
    # label = []
    for i in range(len(data)):
        question = data[i]['question']
        choices = data[i]['choices']
        answer_key = ['A','B','C','D','E'].index(data[i]['answerKey'])

        labels.append(answer_key)
        this_k = k[i]
        choice_len = 0
        thiscontext  = []
        for item in choices:
            # print(item)
            for j in range(k_num):
                if mode == 'gpt':
                    predict_context = this_k[j] + ' ' + question+ ' ' + item['text']+'.'
                elif mode == 'bert':
                    # print("11111111111111111111111111111111")
                    # 需要先用空格分隔成list之后 用list长度计算
                    k_list = this_k[j].split()
                    q_list = question.split()
                    c_list = item['text'].split()
                    # print(len(k_list)+len(q_list)+len(c_list))
                    tmp_l = len(k_list)+len(q_list)+len(c_list)
                    if tmp_l>400:
                        print(tmp_l)
                    if len(k_list)+len(q_list)+len(c_list)>400:
                        predict_context = ' '.join(this_k[:400-len(q_list)-len(c_list)])+' '+ question+' [SEP] ' + item['text']+'.'
                        # print(predict_context)
                        # exit()
                    else:
                        predict_context = this_k[j] + ' '+ question+' [SEP] ' + item['text']+'.'

                thiscontext.append(predict_context)
        contexts.append(thiscontext)
    return contexts,labels


if __name__=='__main__':
    
    data_dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/train.statement.jsonl'
    data = readCsqa(data_dir)