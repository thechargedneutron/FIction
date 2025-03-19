'''
This code assumes all the WHAM and Detic takes are done and we are ready to put together everything in 3D.
'''

import joblib
import numpy as np
from projectaria_tools.core import calibration, data_provider, mps
import json
from datetime import datetime
from tqdm import tqdm
import pickle
import pandas as pd
import os
import json

from utils import is_point_in_obb, select_anticipation_window

keep_top_K = True

with open("/path/to/Detic/datasets/metadata/lvis_v1_train_cat_info.json") as f:
    lvis_dataset = json.load(f)
lvis_object_indexes = [x['name'] for x in lvis_dataset]

def find_closest_voxel_from_world_loc(world_pos, voxel_centers):
    distances = np.linalg.norm(voxel_centers - world_pos, axis=-1)
    return np.unravel_index(np.argmin(distances), distances.shape)

def add_hand_locations(row, human_mesh, voxel_centers):
    frame_idx = int(row['timestamp'] * 30) # I have checked that the fps is always 30
    left_hand_loc = human_mesh[frame_idx]['verts'][2324]
    right_hand_loc = human_mesh[frame_idx]['verts'][5777]

    left_hand_voxel = find_closest_voxel_from_world_loc(left_hand_loc, voxel_centers)
    right_hand_voxel = find_closest_voxel_from_world_loc(right_hand_loc, voxel_centers)

    # left_hand_distances = np.linalg.norm(voxel_centers - left_hand_loc, axis=-1)
    # left_hand_voxel = np.unravel_index(np.argmin(left_hand_distances), left_hand_distances.shape)
    # right_hand_distances = np.linalg.norm(voxel_centers - right_hand_loc, axis=-1)
    # right_hand_voxel = np.unravel_index(np.argmin(right_hand_distances), right_hand_distances.shape)
    row['left_hand_world_loc'] = left_hand_loc.tobytes()
    row['right_hand_world_loc'] = right_hand_loc.tobytes()
    row['left_hand_voxel'] = left_hand_voxel
    row['right_hand_voxel'] = right_hand_voxel

    return row

def find_closest_indices(t, jsonl_file_path):
    # Read the JSONL file and extract timestamps
    json_timestamps = []
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            json_timestamps.append(data['tracking_timestamp_us'] / 1_000_000.)
    
    # Initialize pointers and prepare to find closest indices
    closest_indices = []
    closest_times = []
    t_index = 0
    n = len(t)
    t_timestamps = []
    for txx in range(len(t)):
        time_str = str(t[txx].tracking_timestamp)
        time_obj = datetime.strptime(time_str, "%H:%M:%S.%f")
        total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1_000_000
        t_timestamps.append(total_seconds)

    
    for json_timestamp in json_timestamps:
        # Move the pointer in t to the closest timestamp
        while t_index < n - 1 and abs(t_timestamps[t_index + 1] - json_timestamp) < abs( t_timestamps[t_index] - json_timestamp):
            t_index += 1
        closest_indices.append(t_index)
        closest_times.append(t_timestamps[t_index])
    
    return closest_indices, closest_times, json_timestamps

with open('/path/to/Detic/all_takes_list.txt') as f:
    all_takes = [x.strip() for x in f.readlines()]
try:
    import sys
    take_idx = int(sys.argv[1])
    take_name = all_takes[take_idx]
except:
    take_name = "iiith_cooking_45_1" #georgiatech_covid_14_4 # "georgiatech_cooking_05_01_2"
    # raise NotImplementedError

# Check if the take_name is not from the test split
take2uid = {}
with open('/path/to/egoexo4d/takes.json') as f:
    takes_metadata = json.load(f)
for idx in range(len(takes_metadata)):
    take2uid[takes_metadata[idx]['take_name']] = takes_metadata[idx]['take_uid']
basedir = "/path/to/egoexo4d/annotations"
with open(os.path.join(basedir, "atomic_descriptions_train.json")) as f:
    train_descriptions = json.load(f)
with open(os.path.join(basedir, "atomic_descriptions_val.json")) as f:
    val_descriptions = json.load(f)
atomic_descriptions = {**train_descriptions['annotations'], **val_descriptions['annotations']}
if take2uid[take_name] not in atomic_descriptions:
    print("Take belongs to test data...")
    exit()

if not os.path.exists(f'/path/to/WHAM_largest_area/{take_name}_largest_area.pkl'):
    print("Largest area file not found, WHAM is not successful...")
    exit()
person_data = joblib.load(f'/path/to/WHAM_largest_area/{take_name}_largest_area.pkl')
# Load the trajectory path and point clouds
trajectory_file = f"/path/to/takes/{take_name}/trajectory/closed_loop_trajectory.csv"

trajectory_data = mps.read_closed_loop_trajectory(trajectory_file)
closest_indices, closest_times, json_timestamps = find_closest_indices(trajectory_data, f'/path/to/egoexo4d/takes/{take_name}/trajectory/online_calibration.jsonl')
device_trajectory = [
    it.transform_world_device.translation()[0] for it in trajectory_data
]
timestamps = [
    it.tracking_timestamp for it in trajectory_data
]
rotation_matrix = [
    it.transform_world_device.to_matrix() for it in trajectory_data
]

# Sample the elements correctly
device_trajectory = [device_trajectory[x] for x in closest_indices]
timestamps = [timestamps[x] for x in closest_indices]
rotation_matrix = [rotation_matrix[x] for x in closest_indices]
print(len(rotation_matrix))


def calculate_voxel_centers(data):
    # min max human
    all_humans = np.vstack([data['human'][x]['verts'] for x in data['human'].keys()])
    max_human = np.max(all_humans, axis=0)
    min_human = np.min(all_humans, axis=0)

    # min max bbs
    max_objects = np.max(data['objects']['rrc'].transpose(0, 2, 1).reshape(-1, 3), axis=0)
    min_objects = np.min(data['objects']['rrc'].transpose(0, 2, 1).reshape(-1, 3), axis=0)

    # print(max_human.shape, min_human, max_objects, min_objects)
    max_overall = np.max(np.vstack([max_human, max_objects]), axis=0)
    min_overall = np.min(np.vstack([min_human, min_objects]), axis=0)

    # now voxelize it into 16x16x16
    # Step 1: Compute the size of each voxel
    grid_size = 16
    voxel_size = (max_overall - min_overall) / grid_size

    # Step 2: Generate the center points for each dimension
    x_centers = np.linspace(min_overall[0] + voxel_size[0] / 2, max_overall[0] - voxel_size[0] / 2, grid_size)
    y_centers = np.linspace(min_overall[1] + voxel_size[1] / 2, max_overall[1] - voxel_size[1] / 2, grid_size)
    z_centers = np.linspace(min_overall[2] + voxel_size[2] / 2, max_overall[2] - voxel_size[2] / 2, grid_size)

    # Step 3: Use meshgrid to generate the 3D grid of centers
    x_grid, y_grid, z_grid = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')

    # Step 4: Stack the grids along a new axis to get shape (16, 16, 16, 3)
    voxel_centers = np.stack([x_grid, y_grid, z_grid], axis=-1)

    return voxel_centers


def transform_mesh(verts, idx_origin=454, idx_z=251, idx_y=6241):
    # Step 1: Set the origin at index 454
    origin = verts[idx_origin]
    
    # Translate all points so that the origin becomes the new origin
    translated_verts = verts - origin
    
    # Step 2: Define the Z-axis (vector from idx_z to idx_origin)
    z_vector = verts[idx_origin] - verts[idx_z]
    z_axis = z_vector / np.linalg.norm(z_vector)
    
    # Step 3: Define the Y-axis (vector from idx_y to idx_origin)
    y_vector = verts[idx_origin] - verts[idx_y]
    y_axis = y_vector / np.linalg.norm(y_vector)

    # Verify the angle between them
    # Compute the dot product
    dot_product = np.dot(z_axis, y_axis)
    # Compute the angle in radians
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clipped to avoid numerical errors
    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)
    # print(f"Angle between y and z is {angle_deg}")
    # exit()
    
    # Step 4: Define the X-axis as the cross product of Y and Z (to maintain right-handedness)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Step 5: Orthogonalize Y-axis (make sure it's perpendicular to X and Z)
    y_axis = np.cross(z_axis, x_axis)
    
    # Step 6: Construct the rotation matrix
    rotation_matrix = np.vstack([x_axis, y_axis, z_axis])  # 3x3 matrix
    
    # Step 7: Apply the transformation (rotation)
    transformed_verts = translated_verts @ rotation_matrix.T
    
    return transformed_verts, rotation_matrix

def correct_gravity(mesh, mesh_up_vector, world_up_vector, fixed_point):
    # Normalize the up vectors
    mesh_up_vector = mesh_up_vector / np.linalg.norm(mesh_up_vector)
    world_up_vector = world_up_vector / np.linalg.norm(world_up_vector)

    # Calculate the rotation axis (cross product of mesh up vector and world up vector)
    rotation_axis = np.cross(mesh_up_vector, world_up_vector)
    
    # If the vectors are already aligned, no need to rotate
    if np.allclose(rotation_axis, 0):
        return mesh

    # Normalize the rotation axis
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # Calculate the rotation angle (dot product between the up vectors)
    angle = np.arccos(np.clip(np.dot(mesh_up_vector, world_up_vector), -1.0, 1.0))

    # Function to create a rotation matrix given axis and angle
    def rotation_matrix(axis, theta):
        axis = axis / np.linalg.norm(axis)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        one_minus_cos = 1 - cos_theta

        # Rodrigues' rotation formula
        return np.array([
            [cos_theta + axis[0] * axis[0] * one_minus_cos,
             axis[0] * axis[1] * one_minus_cos - axis[2] * sin_theta,
             axis[0] * axis[2] * one_minus_cos + axis[1] * sin_theta],
             
            [axis[1] * axis[0] * one_minus_cos + axis[2] * sin_theta,
             cos_theta + axis[1] * axis[1] * one_minus_cos,
             axis[1] * axis[2] * one_minus_cos - axis[0] * sin_theta],
             
            [axis[2] * axis[0] * one_minus_cos - axis[1] * sin_theta,
             axis[2] * axis[1] * one_minus_cos + axis[0] * sin_theta,
             cos_theta + axis[2] * axis[2] * one_minus_cos]
        ])

    # Get the rotation matrix
    R = rotation_matrix(rotation_axis, angle)

    # Translate the mesh so that the fixed point becomes the origin
    translated_mesh = mesh - fixed_point

    # Apply the rotation
    rotated_mesh = np.dot(translated_mesh, R.T)

    # Translate the mesh back to its original position
    final_mesh = rotated_mesh + fixed_point

    return final_mesh

if abs(len(timestamps) - max(person_data, key=lambda x: int(x))) > 10:
    raise ValueError
else:
    if len(timestamps) != max(person_data, key=lambda x: int(x)) + 1:
        print(f"Warning.... {len(timestamps)} vs {max(person_data, key=lambda x: int(x)) + 1}")


compiled_final_data = {'human': {}, 'objects': {}, 'env_voxels': {}}
for frame_idx in tqdm(range(0, len(timestamps))):
    # THERE ARE SOME INCONSISTENCY IN THE TRAJECTORY FILES, BUT THE ERROR is < 10 FRAMES WE SKIP THEM
    if int(frame_idx) >= len(timestamps):
        continue
    assert len(person_data[frame_idx].keys()) <= 1, f"Why are there {len(person_data[frame_idx].keys())} keys..."
    if len(person_data[frame_idx].keys()) == 0:
        continue # No person in this frame
    compiled_final_data['human'][frame_idx] = {'timestamp': timestamps[frame_idx]}
    cam_idx = list(person_data[frame_idx].keys())[0]
    verts = person_data[frame_idx][cam_idx]['verts']  # Get the vertex data for this frame
    pose_params = person_data[frame_idx][cam_idx]['pose'][3:] # First three params are location
    betas = person_data[frame_idx][cam_idx]['betas']

    #### STEP 0: -y is the up vector fo the mesh, see where that point goes in the transformation because we want to preserve the up vector to be the final up vector
    #### STEP 1: Convert the person so that the origin is their left eye and +z is outwards and +y is right to left
    verts, rotation_matrix_mesh = transform_mesh(verts)
    gravity_up_direction = rotation_matrix_mesh @ np.array([0., -1.0, 0.])
    #### STEP 2: Apply the transformation from Aria
    verts_homogeneous = np.vstack([verts, gravity_up_direction])
    verts_homogeneous = np.hstack([verts_homogeneous, np.ones((verts_homogeneous.shape[0], 1))])
    transformed_verts_homogeneous = verts_homogeneous @ rotation_matrix[frame_idx].T
    verts = transformed_verts_homogeneous[:-1, :3]
    transformed_up_vector = transformed_verts_homogeneous[-1, :3]
    ##### STEP 3: Aria rotation makes the people float, apply gravity. Use the fact that +z is the up vector in MPS and -y in WHAM
    # The mesh up vector should be the transformed version of the vector and not -y
    mesh_up_vector = transformed_up_vector - verts[454]
    verts = correct_gravity(verts, mesh_up_vector, np.array([0., 0., 1.]), verts[454])


    # Create a mesh for this person
    compiled_final_data['human'][frame_idx]['upvector_person'] = [verts[454], transformed_up_vector]
    compiled_final_data['human'][frame_idx]['verts'] = verts
    compiled_final_data['human'][frame_idx]['pose_params'] = pose_params
    compiled_final_data['human'][frame_idx]['betas'] = betas
    # Load faces later since it is constant
    compiled_final_data['human'][frame_idx]['device_trajectory'] = device_trajectory[frame_idx]
    compiled_final_data['human'][frame_idx]['aria_direction'] = [(rotation_matrix[frame_idx] @ np.array([0., 0., 0., 1.]))[:3], (rotation_matrix[frame_idx] @ np.array([0., 0., 1., 1.]))[:3]]
    compiled_final_data['human'][frame_idx]['rotation_matrix'] = rotation_matrix[frame_idx][:3, :3]

# Also add objects
with open(f"/path/to/final_object_bbs_with_counts_xy_only/{take_name}_object_bbs_obb.pkl", 'rb') as f:
    object_bbs = pickle.load(f)
# print(type(object_bbs))
# print(object_bbs.keys())
# print(len(object_bbs['object_names']))
# print(object_bbs['object_names'])
# print(object_bbs['object_counts'])
if keep_top_K:
    topK = 30
    # Step 1: Sort the object counts in descending order and get the indices
    top_k_indices = np.argsort(object_bbs['object_counts'])[-topK:][::-1]
    # Step 2: Retrieve the corresponding top K rrc, object names, and counts
    top_k_rrc = object_bbs['rrc'][top_k_indices]
    top_k_object_names = [object_bbs['object_names'][i] for i in top_k_indices]
    top_k_object_counts = [object_bbs['object_counts'][i] for i in top_k_indices]
    print(top_k_object_names)
    # Step 3: Update the object_bbs dict with top K results
    object_bbs['rrc'] = top_k_rrc
    object_bbs['object_names'] = top_k_object_names
    object_bbs['object_counts'] = top_k_object_counts


# print(object_bbs['rrc'].shape)
# print(len(object_bbs['object_names']))
compiled_final_data['objects'] = object_bbs


# Also find the x, y, z coordinate extent and convert it to a voxel grid
# max and min in all sides for object bbs and human verts
voxel_centers = calculate_voxel_centers(compiled_final_data)
print(f"shape of voxel center is {voxel_centers.shape}")
env_voxel = np.zeros((16, 16, 16))
print(env_voxel.shape)

# Add left hand, right hand, belly voxel location
for frame_idx in compiled_final_data['human']:
    curr_verts = compiled_final_data['human'][frame_idx]['verts']
    compiled_final_data['human'][frame_idx]['lower_belly_voxel'] = find_closest_voxel_from_world_loc(curr_verts[1769], voxel_centers)
    compiled_final_data['human'][frame_idx]['left_hand_voxel'] = find_closest_voxel_from_world_loc(curr_verts[2324], voxel_centers)
    compiled_final_data['human'][frame_idx]['right_hand_voxel'] = find_closest_voxel_from_world_loc(curr_verts[5777], voxel_centers)

print("Populating the env voxels...")
for x_ in tqdm(range(voxel_centers.shape[0])):
    for y_ in range(voxel_centers.shape[1]):
        for z_ in range(voxel_centers.shape[2]):
            curr_center = voxel_centers[x_, y_, z_]
            for idx in range(len(object_bbs['object_names'])):
                rrc = object_bbs['rrc'][idx]
                object_name = object_bbs['object_names'][idx]
                parts = object_name.rsplit('-', 1)
                assert len(parts) == 2, f"Why single part of this object: {object_name}, {parts}"
                object_name = parts[0]
                if is_point_in_obb(curr_center, rrc) and object_name not in ['person', 'hands']:
                    obj_idx = lvis_object_indexes.index(object_name)
                    assert obj_idx >= 0, "Why not found?"
                    env_voxel[x_, y_, z_] = obj_idx + 1 # 0 is for no occupancy

compiled_final_data['env_voxels'] = env_voxel
compiled_final_data['voxel_centers'] = voxel_centers

print(compiled_final_data['env_voxels'].shape)

# Also add the narration based labels to close the dataset loop
interaction_basedir = "/path/to/FindingObjectInteractionsFromNarrations/compiled_interactions_v1/"
interactions = pd.read_csv(os.path.join(interaction_basedir, f"{take_name}.csv"))
# Add left and right hand positions
interactions_with_hand_locations = interactions.apply(lambda row: add_hand_locations(row, compiled_final_data['human'], voxel_centers), axis=1)
data_samples = select_anticipation_window(interactions_with_hand_locations)
# Save it and think about viz and then training...
# print(interactions_with_hand_locations.head())
# print(len(data_samples))
# print(type(data_samples))
# print(data_samples.keys())
# print(data_samples[125])
compiled_final_data['orig_interaction_dataset'] = interactions_with_hand_locations
# compiled_final_data['interaction_dataset'] = data_samples

# exit()

with open(f'/path/to/dataset/v1_with_orig_dataset_with_orientation_and_betas_xybbox_top30/{take_name}_humans_objects_interactions.pkl', 'wb') as f:
    pickle.dump(compiled_final_data, f)