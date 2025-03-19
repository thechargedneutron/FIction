import os
import json
import pandas as pd
import transformers
import torch
import sys

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

def check_split_files(take_name, num_frames, processed_files, chunk_size=6000):
    """
    Checks if all split files are present for a given take_name and num_frames.
    
    Parameters:
        take_name (str): The name of the take.
        num_frames (int): The total number of frames.
        processed_files (list): A list of processed files (filenames as strings).
        chunk_size (int): The number of frames per chunk. Default is 6000.
    
    Returns:
        missing_chunks (list): List of missing file names.
    """
    # Calculate the number of chunks
    num_chunks = (num_frames // chunk_size) + (1 if num_frames % chunk_size != 0 else 0)
    
    # Generate expected file names
    expected_files = []
    for i in range(num_chunks):
        start_frame = i * chunk_size
        end_frame = min((i + 1) * chunk_size, num_frames)
        expected_filename = f"{take_name}-{start_frame}-{end_frame}_processed.csv.gz"
        expected_files.append(expected_filename)
    
    # Check if all expected files are present
    missing_chunks = [file for file in expected_files if file not in processed_files]
    
    return missing_chunks


try:
    slurm_idx = int(sys.argv[1])
    with open('/path/to/Detic/all_takes_list_with_slam_camera_names_and_num_frames.txt') as f:
        sentences = [x.strip() for x in f.readlines()]
    takes = [x.split()[0] for x in sentences]
    num_frames = [int(x.split()[-1]) for x in sentences]
    take_name = takes[slurm_idx]
    frames_count = num_frames[slurm_idx]
    print(f"Doing idx: {slurm_idx} and task name: {take_name}...")
except:
    take_name = "iiith_cooking_45_1"
    assert False, "Not supported now..."
    print("Doing default extraction...")

if os.path.exists(f"outputs/{take_name}.txt"):
    with open(f"outputs/{take_name}.txt") as f:
        g = [x.strip() for x in f.readlines()]
    if g[-1] == "DONE!":
        print("This take is already processed...")


######### Check once for all takes ###############
# from tqdm import tqdm
# for slurm_idx in tqdm(range(len(takes))):
#     take_name = takes[slurm_idx]
#     frames_count = num_frames[slurm_idx]
#     detic_basedir = "/path/to/Detic/processed/"
#     split_files = [x for x in os.listdir(detic_basedir) if take_name in x]
#     missing_chunks = check_split_files(take_name, frames_count, split_files)
#     if len(missing_chunks) > 0:
#         print(f"take name {take_name} has missing files...")
###################################################


basedir = "/datasets01/egoexo4d/v2/annotations"

take2uid = {}
with open('/datasets01/egoexo4d/v2/takes.json') as f:
    takes = json.load(f)
for idx in range(len(takes)):
    take2uid[takes[idx]['take_name']] = takes[idx]['take_uid']

narrations = []
with open(os.path.join(basedir, "atomic_descriptions_train.json")) as f:
    train_descriptions = json.load(f)
with open(os.path.join(basedir, "atomic_descriptions_val.json")) as f:
    val_descriptions = json.load(f)
atomic_descriptions = {**train_descriptions['annotations'], **val_descriptions['annotations']}
assert take2uid[take_name] in atomic_descriptions, f"{take_name} not found..."
# print(len(atomic_descriptions[take2uid[take_name]]))
for narration_pass_idx in range(len(atomic_descriptions[take2uid[take_name]])):
    if not atomic_descriptions[take2uid[take_name]][narration_pass_idx]['rejected']:
        for narration_text_idx in range(len(atomic_descriptions[take2uid[take_name]][narration_pass_idx]['descriptions'])):
            narr_d = atomic_descriptions[take2uid[take_name]][narration_pass_idx]['descriptions'][narration_text_idx]
            narrations.append((float(narr_d['timestamp']), narr_d['text']))

# Sort -- no particular use but it's okay
narrations = sorted(narrations, key=lambda x: x[0])

# Load object labels
detic_basedir = "/path/to/Detic/processed/"
split_files = [x for x in os.listdir(detic_basedir) if take_name in x]
missing_chunks = check_split_files(take_name, frames_count, split_files)
assert len(missing_chunks) == 0, f"There are some missing chunks for idx {slurm_idx} and take {take_name}, handle them separately..."
all_objects = []
for split_idx in range(len(split_files)):
    data = pd.read_csv(os.path.join(detic_basedir, split_files[split_idx]))
    all_objects += list(data['object_name'].unique())
all_objects = list(set(all_objects))
all_objects_str = ", ".join(all_objects)

# LLM
prompt = "You are given narrations labeled by human annotators for a video. You are also given a set of object labels as per an object detection vocabulary. Find all instances of object interaction where the person would touch an object and map it to all the synonyms or similar words in the vocabulary. Sentences like 'C looks at the fridge' has no object interaction. Objects like cup, glass can be grouped together. Here are the object labels that you have to use: {}. Answer in this format: \n 1. <rewrite first narration> - answer: (object1, object2) \n 2. <rewrite second narration> - answer: NO INTERACTION \n 3. <rewrite third narration> - answer: NO MATCHING OBJECTS. Use 'NO INTERACTION' and 'NO MATCHING OBJECTS' in cases with no interaction and matching objects, respectively. Here are the numbered narrations: {}"


# Split narrations so that it is not too long
MAX_INPUT_CHAR_LEN = 5000
remaining_length = MAX_INPUT_CHAR_LEN - len(all_objects_str) - len(prompt)
print(remaining_length)

narration_batches = []
narration_str = ""
narration_idx = 1
length_count = 0
for idx in range(len(narrations) + 1):
    if idx == len(narrations) or length_count + len(narrations[idx][1]) > remaining_length:
        narration_batches.append(narration_str)
        # Now clear the counters
        narration_idx = 1
        length_count = 0
        narration_str = ""
    
    if idx < len(narrations):
        narration_str += f"{narration_idx}. {narrations[idx][1]}\n"
        narration_idx += 1
        length_count += len(narrations[idx][1])

# Load Llama3
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

response_data = []
file = open(f'outputs/{take_name}.txt', 'w')
for batch_idx in range(len(narration_batches)):
    message = [
        {"role": "system", "content": "You are a helpful AI assistant. Match the narrations with the object labels that is provided."},
        {"role": "user", "content": prompt.format(all_objects_str, narration_batches[batch_idx])}
    ]

    outputs = pipeline(
        message,
        max_new_tokens=2048,
    )

    print("################### QUESTION START ############################")
    file.write("################### QUESTION START ############################\n")
    print(f"{narration_batches[batch_idx]}\n")
    file.write(f"{narration_batches[batch_idx]}\n")
    print("################### QUESTION END ############################")
    file.write("################### QUESTION END ############################\n")

    print("################### ANSWER START ############################")
    file.write("################### ANSWER START ############################\n")

    file.write(outputs[0]["generated_text"][-1]["content"])
    print(outputs[0]["generated_text"][-1]["content"])

    print("################### ANSWER END ############################")
    file.write("################### ANSWER END ############################\n")

file.write("DONE!")
file.close()
