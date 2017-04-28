import os
import sys

conf_list = [(1, 'noref_nosharp'), (2, 'ref_nosharp'), (6, 'ref_power')]
             
cls_list_1 = ['animal', 'vehicle']
cls_list_2 = ['furniture']

if sys.argv[1] == 'may1':
    cls_list = cls_list_1
elif sys.argv[1] == 'may2':
    cls_list = cls_list_2
    
print(cls_list)
for cls in cls_list:

    
    for conf in conf_list:
        conf_idx = conf[0]
        conf_name = conf[1]

        see_cmd = '/home/tmbao_1995/miniconda3/bin/python main.py --load=413 --get_tf_test --config={} --cls={}'.format(conf_idx, cls)
        os.system(see_cmd)