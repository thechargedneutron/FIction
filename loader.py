import json
import os
import pickle
import numpy as np
from tqdm import tqdm
import random

from torch.utils.data import Dataset
import torch

from utils import axis_angle_to_matrix

class InteractionDataset(Dataset):
    def __init__(self, split='train', observation_time=30., anticipation_time=5., future_time=60., scenarios="all", take_all_interactions=True):
        orig_split = split
        split = 'val' if split == 'test' else split
        self.dataset = []
        self.num_future_interaction_per_episode = 0
        self.num_pose_per_interaction_point = 0
        self.scenario_mapping = {}
        self.observation_time = observation_time
        self.anticipation_time = anticipation_time
        self.future_time = future_time
        self.take_all_interactions = take_all_interactions
        if split == 'train':
            self.random_generator = None
        else:
            self.random_generator = random.Random(0)
        # Fixed paths
        self.dataset_path = "/path/to/v1_with_orig_dataset_with_orientation_and_betas/"
        basedir = "/datasets01/egoexo4d/v2/annotations"
        with open('/full/path/to/data/rgb_feature_paths.json') as f:
            self.video_features_path = json.load(f)

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
        takes = [x.split(f"_humans_objects_interactions_compressed.pkl")[0] for x in os.listdir(self.dataset_path) if '_humans_objects_interactions_compressed.pkl' in x]
        with open(f'/full/path/to/data/{orig_split}_takes.txt') as f:
            orig_split_takes = [x.strip() for x in f.readlines()]

        # end_time = [] # For debugging
        self.human_data = {}
        for take_idx, take_name in tqdm(enumerate(takes)):
            if take_name not in orig_split_takes:
                # pick videos from correct train, val, test set
                continue
            if not (scenarios == "all" or scenarios == take2scenario[take_name]):
                # Skip samples from other scenarios
                continue
            if take2uid[take_name] in descriptions:
                assert take_name in self.video_features_path, f"Take {take_idx} {take_name} not found..."
                with open(os.path.join(self.dataset_path, f"{take_name}_humans_objects_interactions_compressed.pkl"), 'rb') as f2:
                    take_interactions = pickle.load(f2)
                with open(os.path.join(self.dataset_path, f"{take_name}_humans_objects_interactions_compressed_1fps.pkl"), 'rb') as f2:
                    self.human_data[take_name] = pickle.load(f2)
                if 'interaction_dataset' in take_interactions:
                    interaction_data = take_interactions['interaction_dataset']
                else:
                    full_interaction_dataset = take_interactions['orig_interaction_dataset']
                    interaction_data, interaction_data_uncompressed, current_time_dict = self.select_anticipation_window(full_interaction_dataset, future_time=self.future_time)
                for datum_idx in interaction_data:
                    curr_df = interaction_data[datum_idx]
                    curr_df_uncompressed = interaction_data_uncompressed[datum_idx]

                    # Find the starting timestamp as the minimum time - 5 seconds
                    observation_end_time = max(0., current_time_dict[datum_idx] - self.anticipation_time)
                    if observation_end_time < 1.0:
                        # Remove these samples
                        continue
                    # end_time.append(observation_end_time)

                    # Find all ground truth object locations
                    gt_voxels = np.zeros((16, 16, 16))
                    if self.take_all_interactions:
                        for loc in curr_df_uncompressed['right_hand_voxel']:
                            gt_voxels[loc] = 1
                    else:
                        for loc in curr_df['right_hand_voxel']:
                            gt_voxels[loc] = 1

                    touch_loc_to_time_map = self.get_timestamps_for_voxel(gt_voxels, curr_df_uncompressed, return_integer_timestamps=True)
                    touch_loc_to_time_map = {key: [ts for ts in timestamps if int(ts) < len(self.human_data[take_name]['human_1fps'])]
                                                            for key, timestamps in touch_loc_to_time_map.items()} # Removing timestamps beyond the human length
                    touch_loc_to_time_map = {key: timestamps for key, timestamps in touch_loc_to_time_map.items() if timestamps} # Remove empty list
                    if not touch_loc_to_time_map:
                        # No pose interaction found with corresponding human pose
                        continue
                    for pos in touch_loc_to_time_map:
                        assert gt_voxels[pos] == 1, "Check for inconsistency..."
                    # print(take_interactions.keys())
                    # exit()
                    self.num_future_interaction_per_episode += np.count_nonzero(gt_voxels)
                    self.num_pose_per_interaction_point += sum([len(touch_loc_to_time_map[x]) for x in touch_loc_to_time_map])
                    self.dataset.append({
                        'take_name': take_name,
                        'observation_end_time': observation_end_time,
                        'gt_voxels': gt_voxels,
                        'env_voxels': take_interactions['env_voxels'],
                        'touch_loc_to_time_map': touch_loc_to_time_map,
                    })
                    curr_data_idx = len(self.dataset) - 1
                    if take2scenario[take_name] not in self.scenario_mapping:
                        self.scenario_mapping[take2scenario[take_name]] = []
                    self.scenario_mapping[take2scenario[take_name]].append(curr_data_idx)

        print(f"Populated dataset with {len(self.dataset)} samples")
        print(f"Number of poses per interaction point : {self.num_pose_per_interaction_point / self.num_future_interaction_per_episode}")
        print(f"Number of future interaction per episode: {self.num_future_interaction_per_episode / len(self.dataset)}")

    def get_scenario_mapping(self):
        return self.scenario_mapping

    def select_anticipation_window(self, df, timestamp_col='timestamp', future_time=60., observation_time=30, debug_with_prediction=False):
        # Anticipation time is the tau_a in AVT, Rohit Girdhar et al.
        # Function to select a random 1-minute window that doesn't extend beyond the timestamp
        # debug_with_prediction is True only for debugging where we convert the task to prediction rather than anticipation for debugging
        if debug_with_prediction:
            print("[WARNING] This is an incorrect setting. To be used only for debugging...")
        subset_dict = {}
        subset_uncompressed_dict = {} # Does not remove duplicates since we need pose data for ALL interactions
        current_time = {}

        for index, row in df.iterrows():
            timestamp = row[timestamp_col]

            # We need minimum anticipation_time margin before we can start anticipation
            # We already handled anticipation_time
            # if timestamp < anticipation_time:
                # continue

            # Define window start and end based on the random offset
            window_start = timestamp
            window_end = window_start + future_time

            if debug_with_prediction:
                window_end = timestamp
                window_start = max(0., timestamp - observation_time)

            # Select rows where the timestamp lies within the 1-minute window
            window_df = df[(df[timestamp_col] >= window_start) & (df[timestamp_col] <= window_end)]

            # Expand the dataframe to have a new row for each object
            window_df_expanded = window_df.set_index(window_df.columns.difference(['object'], sort=False).tolist())['object'].str.split('@@@@@', expand=True).stack().reset_index(level=-1, drop=True).reset_index().rename(columns={0:'object'})

            # Keep only the first interaction
            window_df_sorted = window_df_expanded.sort_values(by=['object', 'timestamp'])
            window_df_unique = window_df_sorted.drop_duplicates(subset='object', keep='first')
            window_df_unique.reset_index(drop=True, inplace=True)

            # Append the resulting subset dataframe to the dictionary
            subset_dict[index] = window_df_unique
            subset_uncompressed_dict[index] = window_df_sorted
            current_time[index] = timestamp

        return subset_dict, subset_uncompressed_dict, current_time

    def __len__(self):
        return len(self.dataset)

    def get_timestamps_for_voxel(self, ground_truth_grid, df, start_time=None, end_time=None, return_integer_timestamps=False):
        # Pre-filter the dataframe based on the timestamp range
        if start_time is not None or end_time is not None:
            df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

        results = {}
        for x in range(ground_truth_grid.shape[0]):
            for y in range(ground_truth_grid.shape[1]):
                for z in range(ground_truth_grid.shape[2]):
                    if ground_truth_grid[x, y, z] == 1:  # Only for positions where grid has a value of 1
                        # Find matching rows in the filtered dataframe
                        matching_rows = df[df['right_hand_voxel'] == (x, y, z)]
                        # Extract the timestamps into a list
                        if not return_integer_timestamps:
                            timestamp_list = matching_rows['timestamp'].tolist()
                        else:
                            matching_rows.loc[:, 'timestamp'] = matching_rows['timestamp'].astype(int)
                            matching_rows = matching_rows.drop_duplicates(subset=['timestamp'])
                            timestamp_list = matching_rows['timestamp'].tolist()
                        # Assert that there is at least one timestamp associated with this voxel
                        assert len(timestamp_list) > 0, f"No timestamp found for voxel position {(x, y, z)}"

                        # Store the result
                        results[(x, y, z)] = timestamp_list
        return results

    def __getitem__(self, index, return_metadata=False):
        datum = self.dataset[index]
        take_name = datum['take_name']
        observation_end_time = datum['observation_end_time']
        observation_start_time = max(0, observation_end_time - self.observation_time)
        features = torch.load(self.video_features_path[take_name][0])[int(observation_start_time):int(observation_end_time)]
        gt_voxels = datum['gt_voxels'].reshape(-1).astype(int)
        env_voxels = datum['env_voxels'].reshape(-1).astype(int)

        # Pad video features
        visual_mask = torch.zeros(int(self.observation_time), dtype=torch.int)
        if features.shape[0] < self.observation_time:
            padding_length = int(self.observation_time - features.shape[0])
            padding = torch.zeros(padding_length, features.shape[1])
            assert len(features.shape) == 2, "Unknown size"
            features = torch.cat((padding, features), dim=0)
        else:
            padding_length = 0
        visual_mask[padding_length:] = 1

        # Now load humans and objects
        curr_data = self.human_data[take_name]
        human = curr_data['human_1fps']
        person_locations = torch.zeros(int(self.observation_time), 3, dtype=torch.int) # 3 for 3 dimensions
        person_location_mask = torch.zeros(int(self.observation_time), dtype=torch.int)
        person_orientations = torch.zeros(int(self.observation_time), 9) # 9 for 3x3 dimensions
        person_orientations_mask = torch.zeros(int(self.observation_time), dtype=torch.int)
        pose_parameters = torch.zeros(int(self.observation_time), 207)
        pose_parameters_mask = torch.zeros(int(self.observation_time), dtype=torch.int)

        # Fill it backwards since we want padding at the start
        loc_idx = int(self.observation_time) - 1  # Start from the last index
        for time_index in range(int(observation_end_time), int(observation_start_time) - 1, -1):
            if loc_idx < 0:
                break  # We've filled all positions in person_locations
            # The features are 1 pose per second so we can directly index time with location
            if time_index >= 0 and time_index < len(human) and human[time_index] is not None:
                location = human[time_index]['lower_belly_voxel']
                rot_matrix = human[time_index]['rotation_matrix'].reshape(-1)
                pose_parameter = axis_angle_to_matrix(torch.tensor(human[time_index]['pose_params'].reshape(-1, 3))).numpy().reshape(-1)
            else:
                # No human present for this time duration
                loc_idx -= 1
                continue
            person_locations[loc_idx] = torch.tensor(location) + 1 # 0 is for unknown location
            person_location_mask[loc_idx] = 1
            person_orientations[loc_idx] = torch.tensor(rot_matrix)
            person_orientations_mask[loc_idx] = 1
            pose_parameters[loc_idx] = torch.tensor(pose_parameter)
            pose_parameters_mask[loc_idx] = 1
            loc_idx -= 1

        # Prepare dataset for the pose generation pipeline
        pose_gen_timestamps = datum['touch_loc_to_time_map']
        if self.random_generator is not None:
            # Deterministic
            sampled_future_loc = self.random_generator.choice(list(pose_gen_timestamps.keys()))
            sampled_timestamp = int(self.random_generator.choice([x for x in pose_gen_timestamps[sampled_future_loc] if human[int(x)] is not None]))
        else:
            # Random
            sampled_future_loc = random.choice(list(pose_gen_timestamps.keys()))
            sampled_timestamp = int(random.choice([x for x in pose_gen_timestamps[sampled_future_loc] if human[int(x)] is not None]))
        sampled_future_pose = axis_angle_to_matrix(torch.tensor(human[sampled_timestamp]['pose_params'].reshape(-1, 3))).numpy()
        sampled_future_betas = human[sampled_timestamp]['betas']
        future_pose_and_orientation = np.concatenate([human[sampled_timestamp]['rotation_matrix'].astype(sampled_future_pose.dtype).reshape(1, 3, 3), sampled_future_pose], axis=0)

        if return_metadata:
            # Load the uncompressed data containing 'verts' -- heavy operation, do not iterate
            with open(os.path.join(self.dataset_path, f"{take_name}_humans_objects_interactions.pkl"), 'rb') as f:
                full_data = pickle.load(f)
            full_interaction_dataset = full_data['orig_interaction_dataset']
            interaction_to_timestamp = self.get_timestamps_for_voxel(datum['gt_voxels'].astype(int), full_interaction_dataset, observation_end_time + self.anticipation_time, observation_end_time + self.anticipation_time + self.future_time)
            # For viz, we need to return all the metadata that we cannot pass to the trainer
            extra_info = {
                'take_name': take_name,
                'observation_start_time': observation_start_time,
                'observation_end_time': observation_end_time,
                # Send the full verts since you don't know where the intersection is
                'human': full_data['human'],
                'sampled_timestamp': sampled_timestamp,
                'voxel_centers': full_data['voxel_centers'],
                # We also need timestamp mapping of all the interaction points
                'interaction_to_timestamp': interaction_to_timestamp
            }

        return {
            'index': index,
            'visual_features': features,
            'visual_mask': visual_mask,
            'location_voxels': person_locations,
            'location_voxels_mask': person_location_mask,
            'orientations': person_orientations,
            'orientations_mask': person_orientations_mask,
            'pose_features': pose_parameters,
            'pose_features_mask': pose_parameters_mask,
            'env_voxels': env_voxels,
            'sampled_future_loc': sampled_future_loc,
            'sampled_future_betas': sampled_future_betas,
            'labels': {'interaction': gt_voxels, 'pose': future_pose_and_orientation},
            **(extra_info if return_metadata else {})
        }

if __name__ == "__main__":
    for split in ['train', 'val', 'test']:
        dataset = InteractionDataset(split, scenarios="all")
        for x in tqdm(range(0, len(dataset))):
            try:
                dataset.__getitem__(x)
            except Exception as e:
                print(f"Error for split {split}, sample {x}, error: {e}")
            # print(f"Fetched index {x}")