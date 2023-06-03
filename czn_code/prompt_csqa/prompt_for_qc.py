import jsonlines
from tqdm import tqdm
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import numpy as np
def getqc(data_dir):
    questions = []
    # print("111")
    sentences = []
    print("reading")
    with jsonlines.open(data_dir,'r') as f:
        for line in f:
            choices =[]
            # print("q:",line["question"]["stem"])
            question = line["question"]["stem"]
            questions.append(line["question"]["stem"])
            # question (i) choice0 (ii) choice1 (iii) choice2 (iv) choice3 (vi)choice4
            choices = []
            for i in range(5):
                choices.append(line["question"]["choices"][i]["text"])
            # print(type(question))
            # print(type(choice[0]))
            # sentence = 'Select the correct answer among the options provided for the following question. \n'\
            # + question +'\n'\
            # +'Options: (i) '+choices[0] \
            #     +' (ii) '+choices[1] \
            #     +' (iii) '+choices[2] \
            #     +' (iv) '+choices[3] \
            #     +' (vi) '+choices[4]+'\n'\
            #     +'Answer should be (i) or (ii) or (iii) or (iv) or (vi)\n'\
            #     + 'Answer:'     

            # sentence = 'Choose one correct answer among the options provided for the following question.\n'+\
            #     question +'\nOptions: A.'+choices[0] \
            #     +' B.'+choices[1] \
            #     +' C.'+choices[2] \
            #     +' D.'+choices[3] \
            #     +' E.'+choices[4]\
            #     +'\n'\
            #     + 'Answer:'     
            # sentences.append([sentence])

            # sentence = 'Select one correct answer among the options provided for the following question.\n'+\
            #     question +'\n'\
            #     +'i.'+choices[0] \
            #     +' ii.'+choices[1] \
            #     +' iii.'+choices[2] \
            #     +' iv.'+choices[3] \
            #     +' v.'+choices[4]\
            #     +'\n'\
            #     + 'Answer:'     
            # sentences.append([sentence])


            # sentence = 'Select one correct answer among the options provided for the following question.\n'+\
            #     question +' Options:'\
            #     +' iii.'+choices[2] \
            #     +' iv.'+choices[3] \
            #     +' v.'+choices[4]\
            #     +' i.'+choices[0] \
            #     +' ii.'+choices[1] \
            #     +'\n'\
            #     + 'Answer:'     
            # sentences.append([sentence])

            # sentence = 'Select one correct answer among the options provided for the following question.\n'+\
            #     question +' Options:'\
            #     +' iv.'+choices[3] \
            #     +' v.'+choices[4]\
            #     +' i.'+choices[0] \
            #     +' ii.'+choices[1] \
            #     +' iii.'+choices[2] \
            #     +'\n'\
            #     + 'Answer:'     
            # sentences.append([sentence])
            

            sentence = 'Select one correct answer among the options provided for the following question.\n'+\
                question +'\nOptions:'\
                +'\n-'+choices[0] \
                +'\n-'+choices[1]\
                +'\n-'+choices[2] \
                +'\n-'+choices[3] \
                +'\n-'+choices[4] \
                +'\n'\
                + 'Answer:'     
            sentences.append([sentence])

            # sentence = 'Select one correct answer among the options provided for the following question.\n'+\
            #     question +' Options:'\
            #     +' v.'+choices[4]\
            #     +' i.'+choices[0] \
            #     +' ii.'+choices[1] \
            #     +' iii.'+choices[2] \
            #     +' iv.'+choices[3] \
            #     +'\n'\
            #     + 'Answer:'     
            # sentences.append([sentence])
                        

            
            # sentence = 'Select one correct answer among the options provided for the following question.\n'+\
            #     question +' Options: option1.'+choices[0] \
            #     +' option2.'+choices[1] \
            #     +' option3.'+choices[2] \
            #     +' option4.'+choices[3] \
            #     +' option5.'+choices[4]+'\n'\
            #     + 'The answer is :'     
            
        #       )


            #few-shot promt
            options = 'i.'+choices[0]+ ' ii.'+choices[1]+ ' iii.'+choices[2]+' iv.'+choices[3]+ ' v.'+choices[4]
            sentence = 'Question1:The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?\n\
Options:ignore,enforce,authoritarian,yell at or avoid? Answer:ignore\n\
Question2:Sammy wanted to go to where the people were.  Where might he go?\n\
Options:i.race track ii.populated areas iii.the desert iv.apartment v.roadblock\n\
Answer:ii.populated areas\n\
Question3:The fox walked from the city into the forest, what was it looking for?\n\
Options:i.pretty flowers ii.hen house iii.natural habitat iv.storybook v.desk\n\
Answer:iii.natural habitat\n\
Question4:What home entertainment equipment requires cable?\n\
Options:i.radio shack ii.substation iii.cabinet iv.television v.dense forest\n\
Answer:iv.television\n\
Question5:The only baggage the woman checked was a drawstring bag, where was she heading with it?\n\
Options:i.garbage can ii.military iii.jewelry store iv.safe v.airport\n\
Answer:v.airport\n\
Question6:{}\n\
Options:{}\n\
Answer:'.format(question,options)


                # question \
                # +'Options: i.'+choices[0] \
                # +' ii.'+choices[1] \
                # +' iii.'+choices[2] \
                # +' iv.'+choices[3] \
                # +' v.'+choices[4]+'\n'\
                # + 'The answer is :'     




            # print(sentence)
            # sentences.append([sentence])
            # exit()
    
    #write to file
    print("writing")
    with jsonlines.open('./outputfile/promptForQC.json','w') as f: 
        for item in tqdm(sentences):
            f.write(item)
        f.close()

    return sentences

# def readqc()


def show_prompt(prompt_dir):
    with jsonlines.open(prompt_dir , 'r') as f:
        for line in f:
            print(line[0])
            # exit()

def init_model(model_name: str,device: torch.device):
    print('model name: ',model_name)
    """
    Initialize a pre-trained LM
    :param model_name: from MODEL_CLASSES
    :param device: CUDA / CPU device
    :return: the model and tokenizer
    """
    # logger.info(f'Initializing {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # if tokenizer.pad_token is None:
    #     if tokenizer.eos_token is not None:
    #         tokenizer.pad_token = tokenizer.eos_token
    #     else:
    #         tokenizer.pad_token = tokenizer.eos_token = tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_answer(sentence):
    # q c0 c1 c2 c3 c4
    input_text = sentence
    input_ids = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).input_ids
    length = 15
    # print(input_ids.shape[1])
    # exit()
    output = model.generate(input_ids=input_ids, max_length = length+input_ids.shape[1] ,do_sample=True, top_k=0, top_p=1, repetition_penalty=1.2, return_dict=True, use_cache=True)
    # output = model.generate(input_ids=input_ids, max_length = length+input_ids.shape[1] ,do_sample=True, top_k=10, top_p=0.9, repetition_penalty=1.2, return_dict=True, use_cache=True)
    # output = model.generate(input_ids=input_ids,do_sample=True, top_k=10, top_p=0.9, repetition_penalty=1.2, return_dict=True, use_cache=True)
    # print("shape: ",input_ids.shape[1])
    # print('before: ',output) 
    # output = output[0][(input_ids.shape[1]):-1].squeeze(0)
    # print(output)

    #Todo: 把答案截取出来
    res =  tokenizer.batch_decode(output)
    # print(res)   # ['xxx']
    # 截取
    # print("len res,len sentence:",len(res[0]),len(sentence[0]))
    # res_text = res[0][len(sentence[0]):]
    # print('res_text ',res_text)
    # exit()
    # <|endoftext|>
    res_text = res[0][ res[0].find('Answer:')+7:]
    if '<|endof' in res_text:
        # 要在token中找end还是字符串中
        res_text = res_text[: res_text.find('<|endof')]

    # print(res)
    return res_text

def csqaDataset(data_path):
    # 此函数需要把 正确答案放在第一个 其他答案跟在后面
    res = []
    with jsonlines.open(data_path,'r') as f:
        label2id = {'A':0,'B':1,'C':2,'D':3,'E':4}
        for line in f:
            answer_key = line['answerKey']
            # "choices": [{"label": "A", "text": "complete job"}, {"label": "B", "text": "learn from each other"}, {"label": "C", "text": "kill animals"}, {"label": "D", "text": "wear hats"}, {"label": "E", "text": "talk to each other"}]
            choices = line['question']['choices']
            c = []
            for choice in choices:
                c.append(choice['text'])
            idx = label2id[answer_key]
            res.append([c[idx]] + c[:idx]+ c[idx+1:] )
    return res 
def scale_scores(scores, temperature=0.1):
    return torch.exp(scores / temperature)
def evaluate(data_path,supporters_list):
    #supporters_list : 生成的答案

    # supporters_list = pickle.load(open('%s.gpt2xlarge.qa.1.00penalty.topP0.90.minlen2.sample500.pkl' % (args.eval_data_file), 'rb'))
    # for i, supporters in enumerate(supporters_list):
    #     supporters_list[i] = [upper_first_word(supporter) for supporter in supporters[:500]]

    
    # data_path = ''
    dataset = csqaDataset(data_path)
    model_name_or_path = '/users5/znchen/Semantic-based-QA-main/models/sentence-robert-large-nli-mean-tokens'
    embedder = SentenceTransformer(model_name_or_path, device=device)

    acc = []
    num = -1
    for options, supporters in tqdm(list(zip(dataset, supporters_list))):
        num = num+1
        # if num<160:
        #     continue
        # if num>165:
        #     exit()
        # embeddings = embedder.encode(options + supporters, convert_to_tensor=True)
        # option_embeddings = embeddings[:len(options)]
        # supporter_embeddings = embeddings[len(options):]
        # print(options)
        
        option_embeddings = embedder.encode(options, convert_to_tensor=True)
        # if supporters==[]:
        #     print(supporters)
        supporter_embeddings = embedder.encode(supporters, convert_to_tensor=True)
        scores = []
        for option_embedding in option_embeddings:
            score = scale_scores(util.pytorch_cos_sim(option_embedding, supporter_embeddings)[0], temperature=0.1).mean().item()
            scores.append(score)
        acc.append(float(scores[0] > max(scores[1:])))
    print('Accuracy:', np.mean(acc))


def seqa_generate(model, tokenizer, prompt, repetition_penalty=1., top_k=0, top_p=1, min_length=2, data_type='CSQA'):
    '''
    encoded_prompt: [1, docuemnt_length]
    prompt: str sentence
    '''
    encoded_prompt = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).input_ids
    filtered_output_sequences = []
    prefix = tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)   
    # prefix ?
    max_length = 15
    if data_type == 'CosmosQA':
        max_length = 20
    
    for generation_round in range(500):
        output_sequences = model.generate(
            input_ids=encoded_prompt.to(device),
            max_length=max_length + len(encoded_prompt[0]),
            temperature=1.,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            num_return_sequences=100, # If out-of-memory, use a smaller num_return_sequences.
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        texts = [tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)[len(prefix) : ] for generated_sequence in output_sequences]
        for text, generated_sequence in zip(texts, output_sequences):
            if len(text.strip().split(' ')) < min_length:
                continue
            if '\n' in text:
                continue
            if data_type == 'SocialIQA':
                if '<|endoftext|>' in text:
                    filtered_output_sequences.append(generated_sequence)
                elif '.' in text:
                    filtered_output_sequences.append(generated_sequence)
                else:
                    continue
            else:
                if '<|endoftext|>' in text and '.' in text[: text.find('<|endoftext|>')]:
                    filtered_output_sequences.append(generated_sequence)
                elif '<|endoftext|>' not in text and '.' in text:
                    filtered_output_sequences.append(generated_sequence)
                else:
                    continue

        print(len(filtered_output_sequences))
        if len(filtered_output_sequences) > 500:
            break

    generated_sequences = []
    entire_texts = []

    for generated_sequence in filtered_output_sequences:
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[len(prefix) : ]

        if '<|endoftext|>' in text:
            text = text[: text.find('<|endoftext|>')].strip()
        else:
            text = text.strip()
        if data_type == 'SocialIQA':
            if '.' in text:
                text = text[: text.find('.') + 1].strip()
        else:
            text = text[: text.find('.') + 1].strip()
        
        entire_texts.append('%s %s' % (prefix, text))

        generated_sequences.append(text)

    print('Non-repetitive voter number:', len(set(generated_sequences)))

    return entire_texts, generated_sequences



# device = 'cpu'
device = 0
device = torch.device(f'cuda:{device}') if device >= 0 else torch.device("cpu")

model, tokenizer = init_model('gpt2-large',device)

if __name__=='__main__':


    data_dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/dev.statement.jsonl'
    sentences = getqc(data_dir)

    # prompt_dir = './outputfile/promptForQC.json'
    # show_prompt(prompt_dir)

    # mysentence = 'What to eat for lunch? Canteen or go out?'
    # res = generate_answer(mysentence)
    # print(res[0])
    # exit()
    answers  = []
    num = 0
    for sentence in sentences:
        # answers = []
        num += 1
        print(num)
        # if num%5==1:
        #     print("question "+ str(int(num/5)))

        entire_texts, generated_sequences = seqa_generate(model, tokenizer, sentence, repetition_penalty=1., top_k=0, top_p=1, min_length=2, data_type='CSQA')
        # print(entire_texts, generated_sequences)
        # for i in range(5):
            # res = generate_answer(sentence)
            # # print(res[0][len(sentence):-1])
            # if len(res.split(' '))>2:
            #     answer.append(res.strip())
            # # print('、、、、、、、、、')
        # for item in answer:
        #     print(item)
        # print('-------------------------')
        # res = generate_answer(sentence)
        # print(res[0]) 
        # print(' ')
        answers.append(generated_sequences)
        
        # if num >=1:
        #     break
    # save_path = './data/csqa_dev_answer.pkl'
    # pickle.dump(answers, open(save_path, 'wb'))
    
    with jsonlines.open('./outputfile/answers.json','w') as f:
        for item in answers:
            f.write(item)
    
    with jsonlines.open('/users5/znchen/Question2Knowledge/SearchQasP/outputfile/answers.json','r' )as f:
        answers = []
        for line in f:
            answer = []
            for ans in line:
                # print('before: ',ans)
               
                ans = ans.strip()
                print(ans)
                answer.append(ans)
                # print("after: ",ans)

            answers.append(answer)
    supporters_list = answers
    data_dir = '/users5/znchen/Question2Knowledge/SearchQasP/data/csqa/statement/dev.statement.jsonl'
    evaluate(data_dir,supporters_list)
