import h5py

count_file = h5py.File('full_count_voc.hdf5', 'a')

check = {}
check['aeroplane'] = 2
check['bicycle'] = 2
check['boat'] = 2
check['bottle'] = 1
check['car'] = 2
check['chair'] = 1
check['cow'] = 0
check['diningtable'] = 1
check['dog'] = 0
check['horse'] = 0
check['pottedplant'] = 1
check['sheep'] = 0
check['train'] = 2
check['tvmonitor'] = 1

count_dset = []

for key, value in check.items():
    count_dset.append(count_file[key])

animal_count_dset = count_file.create_dataset('animal', (27088,), dtype='i')
furniture_count_dset = count_file.create_dataset('furniture', (27088,), dtype='i')
vehicle_count_dset = count_file.create_dataset('vehicle', (27088,), dtype='i')

for i in range(27088):
    count_animal = 0
    count_furniture = 0
    count_vehicle = 0

    for key, value in check.items():
        if value == 0:
            count_animal += count_file[key][i]
        elif value == 1:
            count_furniture += count_file[key][i]
        elif value == 2:
            count_vehicle += count_file[key][i]
            
        animal_count_dset[i] = count_animal
        furniture_count_dset[i] =  count_furniture
        vehicle_count_dset[i] = count_vehicle
        
count_file.close()

