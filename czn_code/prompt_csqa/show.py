import jsonlines
from tqdm import tqdm
data_dir = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/gened_by_key_word_prompt.txt'
num = 0
# with jsonlines.open(data_dir,'r') as f:
#     for line in f:
#         if num%10 ==5:
#             sentence= line[0]
#             print(sentence)
#             print("----------------")
#         num += 1 

def readk(data_dir):
    contexts = []
    with jsonlines.open(k_dir,'r') as f:
        for line in f:
            contexts.append(line)
    show_dir = './show.jsonl'
    with jsonlines.open(show_dir,'w') as f:
            for context in contexts:
            # print('-------------------------')
                f.write('-------------------------')
                f.write(context['data:']['question'])
            # print(line['data:']['question'])
                k = context['knowledge: ']
                for item in k:
                # print(item)
                    f.write(str(item))
    return     

if __name__=='__main__':
    k_dir = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/k_gen_by_qkk.jsonl'
    readk(k_dir)

    # a,b,c,d,e = 0,0,0,0,0
    # data_dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/dev.statement.jsonl'
    # with jsonlines.open(data_dir,'r') as f:
    #     l = 0
    #     for line in tqdm(f):
    #         l+=1
            
    #         # answer = line["answerKey"]
    #         # if answer=='A':
    #         #     a +=1
    #         # elif answer=='B':
    #         #     b+=1
    #         # elif answer=='C':
    #         #     c+=1
    #         # elif answer=='D':
    #         #     d+=1
    #         # else:
    #         #     e+=1

    #         #统计一下答案长度
    #         max_len = 0
    #         answers = line["question"]["choices"]
    #         for answer in answers:
    #             # print(answer["text"].split(' '))
    #             # exit()
    #             l = len(answer["text"].split(' '))
    #             if l>max_len:
    #                 max_len = l
            

    #         # if line['question']['stem']==
    # # print(a/l,b/l,c/l,d/l,e/l)
    # print(max_len)

    # # 把/users5/znchen/Question2Knowledge/SearchQasP/outputfile/answers.json中的答案从answer:处开始截取
    # with jsonlines.open('/users5/znchen/Question2Knowledge/SearchQasP/outputfile/answers.json','r' )as f:
    #     answers = []
    #     for line in f:
    #         answer = []
    #         for ans in line:
    #             print('before: ',ans)
    #             ans = ans[ans.find('Answer:')+7:].strip()
    #             answer.append(ans)
    #             print("after: ",ans)
    #         exit()
    #         answers.append(answer)
            
                