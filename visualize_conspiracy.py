import sys
import os

sofa_list = [3811,3343,4028,228,2063,2229,1965,1691,4029,1869,2097,3998]
bus_list = [2451,702,6,1252,2451,1252,2794,2740,1387,3752,648,2717]
cat_list = [2041,68,3818,3780,2162,1812,2523,3560,2041,2066,669,778]
motorbike_list =[3672,2785,1155,1555,2612,1307,1676,2198,132,140,1698,39]
bird_list = [3215,669,665,22,3978,3111,3441,3682,219,1397,428,2936]
person_list = [3464,2941,3845,3030,3823,3359,3694,4034,2579,3390,2705,814]

may1_dict = {'sofa':sofa_list, 'bus':bus_list, 'cat':cat_list}
may2_dict = {'motorbike': motorbike_list, 'bird':bird_list, 'person':person_list}

if sys.argv[1] == 'may1':
    cls_dict = may1_dict
elif sys.argv[1] == 'may2':
    cls_dict = may2_dict
    

for cls_name, cls_list in cls_dict.items():
    conf_idx = 1
    while conf_idx < 7:
        img_id_list = [cls_list[(conf_idx-1)*2], cls_list[(conf_idx-1)*2+1]]
    
        for img_id in img_id_list:
            vi_cmd = '/home/tmbao_1995/miniconda3/bin/python main.py --load=206 --see_test_idx={} --config={} --cls={}'.format(img_id, conf_idx, cls_name)
            os.system(vi_cmd)
            
        conf_idx += 1
        
