import sys
import os

animal_A_list = [2041, 3998, 4013, 3764, 9, 2955, 4013, 3673, 3818, 3558, 3023, 3115]
vehicle_A_list = [999, 2480, 640, 1392, 3317, 3019, 3917, 3800, 3141, 2570, 3777, 3583]
indoor_A_list = [3308, 3805, 3739, 3668, 3464, 2971, 2392, 3118, 3308, 3478, 2834, 3943]

conf_list_1 = [(0, 'no_attention'), (1, 'noref_nosharp'), (2, 'ref_nosharp'), (3, 'noref_softmax')]
conf_list_2 = [(4, 'ref_softmax'), (5, 'noref_power'), (6, 'ref_power')]
if sys.argv[1] == 'may1':
    conf_list = conf_list_1
elif sys.argv[1] == 'may2':
    conf_list = conf_list_2
    
cls_dict = {'animal_A':animal_A_list, 'vehicle_A':vehicle_A_list, 'indoor_A':indoor_A_list}
cls_list = ['animal_A', 'vehicle_A', 'indoor_A']

i = 0
for cls in cls_list:
    img_idx_list = cls_dict[cls]
    for conf in conf_list:
        conf_idx = conf[0]
        img_id_list = [img_idx_list[(conf_idx-1)*2], img_idx_list[(conf_idx-1)*2+1]]
        
        for img_id in img_id_list:
            vi_cmd = '/home/tmbao_1995/miniconda3/bin/python main.py --load=206 --see_test_idx={} --config={} --cls={} --save_idx={}'.format(img_id, conf_idx, cls_name, i)
            os.system(vi_cmd)
            i += 1
        