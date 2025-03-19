'''
Files with the name: utokyo_pcr_2001_35_4_humans_objects_interactions.pkl is very heavy
as it has humans and objects per frame. we can remove that for the init of the dataloader
'''

import os
import pickle
import numpy as np
from multiprocessing import Pool

basedir = '/path/to/dataset/v1_with_orig_dataset_with_orientation_and_betas/'

files_to_process = [x for x in os.listdir(basedir) if '_humans_objects_interactions.pkl' in x]

def sample_data_per_second(data_dict):
    keys = sorted(data_dict.keys())  # Sort the keys to ensure they are in order
    samples = []

    # Divide keys into groups of 30
    for i in range(0, len(keys), 30):
        group = [key for key in keys if i <= key < i + 30]  # Get the group within 30-frame intervals
        # assert group, f"No data found in group for frames {i} to {i + 29}"  # Assert that group is not empty
        if group:
            min_key = min(group)  # Select the minimum key from the group
            samples.append(data_dict[min_key])  # Append the corresponding data to the samples list
        else:
            samples.append(None) # cases where no human is present in those frames

    return samples

add_1fps_feats = True

# for file in files_to_process:

def process_file(file):
    suffix = ".pkl" if not add_1fps_feats else "_1fps.pkl"
    print(file)
    if os.path.exists(f'{basedir}{file[:-4]}_compressed{suffix}'):
        return
    with open(f'{basedir}{file}', 'rb') as f:
        data = pickle.load(f)
    print(data.keys())

    if add_1fps_feats:
        [inner_dict.pop('verts', None) for inner_dict in data['human'].values()]
        data['human_1fps'] = sample_data_per_second(data['human'])
        del data['orig_interaction_dataset']

    del data['human']
    del data['objects']
    del data['voxel_centers']

    with open(f'{basedir}{file[:-4]}_compressed{suffix}', 'wb') as f:
        pickle.dump(data, f)

with Pool(processes=8) as pool:
    # Map the list of files to the process_file function
    pool.map(process_file, files_to_process)
