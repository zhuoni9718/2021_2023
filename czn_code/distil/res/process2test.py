import jsonlines

def transfer(rationale_dir,test_data_dir,test_dir):
    data,rationale = [],[]
    with jsonlines.open(test_data_dir,'r') as f:
        for line in f:
            data.append(line['data'])
        
    with open(rationale_dir,'r') as f:
        for line in f.readlines():
            rationale.append(line.strip('\n').strip('\"'))

    with jsonlines.open(test_dir,'w') as f:
        for i in range(len(data)):
            f.write({'data':data[i],'k':rationale[i]})
    print('writed')

if __name__=='__main__':
    rationale_dir='/users5/znchen/distil/res/t5large/r8_train.res'
    test_data_dir='/users5/znchen/distil/input/csqa/train.gpt.jsonl'
    test_dir=rationale_dir+'.r0.jsonl'
    transfer(rationale_dir,test_data_dir,test_dir)

    # train_data_dir = '/users5/znchen/distil/input/csqa/train.gpt.jsonl'
    # train_rationale_dir = '/users5/znchen/distil/res/t5large/R0test.res'
    # test_dir=train_rationale_dir+'.r0.jsonl'
    # transfer(train_rationale_dir,train_data_dir,test_dir)