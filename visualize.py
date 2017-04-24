import sys
import os

sofa_list = [3127, 3017, 3358, 1247, 3943, 3751]
bus_list = [138, 1257, 2978, 3296, 138, 2355]
cat_list = [1714, 3756, 3464, 3674, 1867, 3445]
motorbike_list =[3015, 2357, 3802, 3750, 2366, 4011]
bird_list = [2289, 1174, 3069, 955, 2771, 1344]
person_list = [1691, 3429, 4051, 3958, 1811, 3987]

may1_dict = {'sofa':sofa_list, 'bus':bus_list, 'cat':cat_list}
may2_dict = {'motorbike': motorbike_list, 'bird':bird_list, 'person':person_list}

if sys.argv[1] == 'may1':
    cls_dict = may1_dict
elif sys.argv[1] == 'may2':
    cls_dict = may2_dict
    
for cls_name, cls_list in cls_dict.items():
    conf_idx = 1
    while conf_idx < 7:
        img_id_list = [cls_list[(conf_idx-1)]]
    
        for img_id in img_id_list:
            vi_cmd = '/home/tmbao_1995/miniconda3/bin/python main.py --load=206 --see_test_idx={} --config={} --cls={}'.format(img_id, conf_idx, cls_name)
            os.system(vi_cmd)
            
        conf_idx += 1