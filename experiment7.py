import os
import sys

cls_list = ['animal_A', 'vehicle_A', 'indoor_A']

conf_list_1 = [(0, 'no_attention'), (1, 'noref_nosharp'), (2, 'ref_nosharp'), (3, 'noref_softmax')]
conf_list_2 = [(4, 'ref_softmax'), (5, 'noref_power'), (6, 'ref_power')]
if sys.argv[1] == 'may1':
    conf_list = conf_list_1
elif sys.argv[1] == 'may2':
    conf_list = conf_list_2
    
print(conf_list)
for cls in cls_list:

    
    for conf in conf_list:
        conf_idx = conf[0]
        conf_name = conf[1]
        train_cmd = '/home/tmbao_1995/miniconda3/bin/python main.py --epoch=9 --lr=1e-3 --config={} --cls={}'.format(conf_idx, cls)
        os.system(train_cmd)
        
        f = open('backup/checkpoint', 'r')
        line = f.readlines()[0]
        num = line.split('-')[1]
        num = int(num[:-2])
        train_cmd = '/home/tmbao_1995/miniconda3/bin/python main.py --epoch=18 --lr=1e-4 --config={} --cls={} --load={}'.format(conf_idx, cls, num)
        os.system(train_cmd)
        f.close()
    
        os.system('rm *.jpg')
        f = open('backup/checkpoint', 'r')
        line = f.readlines()[0]
        num = line.split('-')[1]
        num = int(num[:-2])
        print('Visualizing')
        print(num, conf_idx, cls)
        see_cmd = '/home/tmbao_1995/miniconda3/bin/python main.py --load={} --see_test --config={} --cls={}'.format(num, conf_idx, cls)
        os.system(see_cmd)
        f.close()
        
        if conf_idx != 0:
            zip_cmd = 'zip {}_{}_A.zip *.jpg'.format(conf_name, cls)
            os.system(zip_cmd)
        
        os.system('rm *.jpg')
    
