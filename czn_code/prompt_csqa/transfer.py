import json
import jsonlines

def dataTransferTok(k_dir):
    k = []
    with jsonlines.open(k_dir,'r') as f:
        for line in f:
            if 'knowledge: ' in line:
                k.append(line["knowledge: "])
            else:
                k.append(line["knowledge:"])
    write_dir = k_dir[:-4]+'_transfered.jsonl'
    with jsonlines.open(write_dir,'w') as f2:
        for item in k:
            f2.write(item)
    return write_dir



if __name__=='__main__':
    k_dir = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/0220/k_gen_by_fixed_Q2K0219.txt'
    # k_dir = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/0220/k_gen_by_fixed_qk.txt'
    # promptnames = ["logic_abs","logic_con","logic_ana","logic_ind","logic_cau",'logic_comprehensive']
    # promptnames = ['logic_abs']
    # data_dir = "/users5/znchen/MHGRN-master/MHGRN-master/data/csqa/statement/dev.statement.jsonl"
    # for item in promptnames:
    #     promptname = item
    #     k_dir = './k_gen_by_'+promptname+'_512.jsonl'
    #     dataTransferTok(k_dir)
    dataTransferTok(k_dir)