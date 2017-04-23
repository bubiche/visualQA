import os
import sys

conf_list = [(0, 'no_attention'), (1, 'noref_nosharp'), (2, 'ref_nosharp'),
             (3, 'noref_softmax'), (4, 'ref_softmax'), (5, 'noref_power'),
             (6, 'ref_power')]
             
cls_list_1 = ['sofa', 'bus', 'cat']
cls_list_2 = ['motorbike', 'bird', 'person']

if sys.argv[1] == 'may1':
    cls_list = cls_list_1
elif sys.argv[1] == 'may2':
    cls_list = cls_list_2
    
print(cls_list)
for cls in cls_list:

    
    for conf in conf_list:
        conf_idx = conf[0]
        conf_name = conf[1]

        see_cmd = '/home/tmbao_1995/miniconda3/bin/python main.py --load=620 --see_test --config={} --cls={}'.format(conf_idx, cls)
        os.system(see_cmd)
        
        if conf_idx != 0:
            zip_cmd = 'zip {}_{}_A_full.zip *.jpg'.format(conf_name, cls)
            os.system(zip_cmd)
        
        os.system('rm *.jpg')
    
