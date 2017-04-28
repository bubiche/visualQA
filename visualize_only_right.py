import sys
import os

conf_list = [(1, 'noref_nosharp'), (2, 'ref_nosharp'), (6, 'ref_power')]
             
cls_list_1 = ['dog', 'horse', 'vehicle']
cls_list_2 = ['pottedplant', 'aeroplane', 'furniture']

img_idx = {'vehicle':2620, 'aeroplane':1463, 'dog':1999, 'pottedplant':3128, 'horse': 3092, 'furniture':12}

if sys.argv[1] == 'may1':
    cls_list = cls_list_1
elif sys.argv[1] == 'may2':
    cls_list = cls_list_2
    
print(cls_list)
for conf in conf_list:
    conf_idx = conf[0]
    for cls in cls_list:
        img = img_idx[cls]
        vi_cmd = '/home/tmbao_1995/miniconda3/bin/python main.py --load=620 --see_test_idx={} --config={} --cls={}'.format(img, conf_idx, cls)
        os.system(vi_cmd)