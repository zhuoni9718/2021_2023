import jsonlines

cg_dir = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/cg/k_ft_t5_cg.train.jsonl.res'
# cg_dir = '/users5/znchen/Question2Knowledge/SearchQasP/outputfile/dif_model1/cg/k_ft_t5_cg_test.jsonl.res'
cgk = []
cont=0
with jsonlines.open(cg_dir,'r') as f:
    for line in f:
        if cont<1221:
            cont+=1
            continue
            
        cgk.append(line['knowledge:'])
r_dir = '/users5/znchen/distil/res/t5large/r8_train.res.r0.jsonl'
# r_dir = '/users5/znchen/distil/res/t5large/R8test.res.r0.jsonl'
data = []

with jsonlines.open(r_dir,'r') as f:
    for line in f:
        # print(line)
        # exit()
        data.append(line)

for i in range(len(data)):
    data[i]['cgk'] = cgk[i]
write_dir = './cgkr/t5train.jsonl'
# write_dir = './cgkr/t5dev.jsonl'
with jsonlines.open(write_dir,'w') as f:
    for item in data:
        f.write(item)
    print(f'write to{write_dir}')