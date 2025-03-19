import os
import json
from tqdm import tqdm
import random
random.seed(0)

split = 'val'

# Fixed paths
dataset_path = "/path/to/dataset/v1_with_orig_dataset_with_orientation/"
basedir = "/datasets01/egoexo4d/v2/annotations"
with open('/path/to/Training/data/rgb_feature_paths.json') as f:
    video_features_path = json.load(f)

# First differentiate between train and val set
with open(os.path.join(basedir, f"atomic_descriptions_{split}.json")) as f:
    descriptions = json.load(f)['annotations']
take2uid = {}
take2scenario = {}
with open('/datasets01/egoexo4d/v2/takes.json') as f:
    takes = json.load(f)
for idx in range(len(takes)):
    take2uid[takes[idx]['take_name']] = takes[idx]['take_uid']
    take2scenario[takes[idx]['take_name']] = takes[idx]['parent_task_name']
takes = [x.split(f"_humans_objects_interactions_compressed.pkl")[0] for x in os.listdir(dataset_path) if '_humans_objects_interactions_compressed.pkl' in x]

# end_time = [] # For debugging
valid_take_lists = []
for take_idx, take_name in tqdm(enumerate(takes)):
    if take2uid[take_name] in descriptions:
        valid_take_lists.append(take_name)

print(f"First take before: {valid_take_lists[0]}")
random.shuffle(valid_take_lists)
print(f"First take after: {valid_take_lists[0]}")

val_takes = valid_take_lists[:len(valid_take_lists) // 2]
test_takes = valid_take_lists[len(valid_take_lists) // 2:]

print(f"Total size of val split is {len(val_takes)}")
print(f"Total size of test split is {len(test_takes)}")
# with open('../data/train_takes.txt', 'w') as f:
#     for take_name in valid_take_lists:
#         f.write(f"{take_name}\n")
with open('../data/val_takes.txt', 'w') as f:
    for take_name in val_takes:
        f.write(f"{take_name}\n")
with open('../data/test_takes.txt', 'w') as f:
    for take_name in test_takes:
        f.write(f"{take_name}\n")
print("Splitting ")