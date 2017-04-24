import sys
import os

vehicle_A_list = [3592, 3132, 2650, 1619, 2208, 2858]
indoor_A_list = [2450, 1691, 3590, 3751, 2651, 3357]

conf_list_1 = [(1, 'noref_nosharp'), (2, 'ref_nosharp'), (3, 'noref_softmax')]
conf_list_2 = [(4, 'ref_softmax'), (5, 'noref_power'), (6, 'ref_power')]
if sys.argv[1] == 'may1':
    conf_list = conf_list_1
elif sys.argv[1] == 'may2':
    conf_list = conf_list_2
    
cls_dict = {'vehicle_A':vehicle_A_list, 'indoor_A':indoor_A_list}
cls_list = ['vehicle_A', 'indoor_A']

i = 0
for cls in cls_list:
    img_idx_list = cls_dict[cls]
    for conf in conf_list:
        conf_idx = conf[0]
        img_id_list = [img_idx_list[(conf_idx-1)]]
        
        for img_id in img_id_list:
            vi_cmd = '/home/tmbao_1995/miniconda3/bin/python main.py --load=206 --see_test_idx={} --config={} --cls={} --save_idx={}'.format(img_id, conf_idx, cls, i)
            os.system(vi_cmd)
            i += 1
        
        
