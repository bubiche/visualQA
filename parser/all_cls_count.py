import h5py

count_file = h5py.File('full_count_voc.hdf5', 'a')

check = {}
check['sofa'] = 2
check['bus'] = 1
check['cat'] = 0
check['motorbike'] = 1
check['bird'] = 0
check['person'] = 2

count_dset = []

for key, value in check.items():
    count_dset.append(count_file[key])

animal_count_dset = count_file.create_dataset('animal_A', (27088,), dtype='i')
furniture_count_dset = count_file.create_dataset('vehicle_A', (27088,), dtype='i')
vehicle_count_dset = count_file.create_dataset('indoor_A', (27088,), dtype='i')

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
    
    if i%100 == 0: print(i)
        
count_file.close()

