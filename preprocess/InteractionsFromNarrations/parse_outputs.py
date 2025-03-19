import re
import Levenshtein
import pandas as pd
import json
import os

# Load object labels
detic_basedir = "/path/to/Detic/processed/"
basedir = "/datasets01/egoexo4d/v2/annotations"

take2uid = {}
with open('/datasets01/egoexo4d/v2/takes.json') as f:
    takes = json.load(f)
for idx in range(len(takes)):
    take2uid[takes[idx]['take_name']] = takes[idx]['take_uid']

with open(os.path.join(basedir, "atomic_descriptions_train.json")) as f:
    train_descriptions = json.load(f)
with open(os.path.join(basedir, "atomic_descriptions_val.json")) as f:
    val_descriptions = json.load(f)
atomic_descriptions = {**train_descriptions['annotations'], **val_descriptions['annotations']}


def are_similar(text1, text2, threshold=0.8):
    # Calculate Levenshtein distance and normalize by the length of the longer text
    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return True  # Both texts are empty
    similarity = 1 - Levenshtein.distance(text1, text2) / max_len
    return similarity >= threshold, similarity

def parse_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()

        # Define the pattern for blocks of questions and answers
        block_pattern = re.compile(
            r"################### QUESTION START ############################\s*(.*?)"
            r"################### QUESTION END ############################\s*"
            r"################### ANSWER START ############################\s*(.*?)"
            r"################### ANSWER END ############################",
            re.DOTALL
        )

        # Find all blocks
        blocks = block_pattern.findall(content)
        if not blocks:
            raise ValueError("No valid question-answer blocks found.")
        
        assert all([len(blocks[x]) == 2 for x in range(len(blocks))]), "Some issue with the formatting..."

        parsed_data = []

        for question_block, answer_block in blocks:
            # Parse questions
            question_pattern = re.compile(r"(\d+)\.\s*(.*?)\s*(?=\d+\.|$)", re.DOTALL)
            questions = question_pattern.findall(question_block)
            # print(questions)

            # Parse answers
            # answer_pattern = re.compile(r"(\d+)\.\s*(.*?)\s*-\s*answer:\s*(.*?)\s*(?=\d+\.|$)", re.DOTALL)
            answer_pattern = re.compile(r"(\d+)\.\s*(.*?)\s*-?\s*answer:\s*(.*?)\s*(?=\d+\.|$)", re.DOTALL)
            answers = answer_pattern.findall(answer_block)
            # print("X"*100)
            # print(answers, questions)
            if len(questions) == 1 and questions[0][1] == '':
                # One special case in upenn_0710_Cooking_1_3.txt
                questions = []
            # exit()
            if not (len(questions) == 0 and len(answers) == 0) and not questions:
                raise ValueError("Questions parsing failed.")
            if not (len(questions) == 0 and len(answers) == 0) and not answers:
                raise ValueError("Answers parsing failed.")

            # Check if the number of questions matches the number of answers
            if len(questions) != len(answers):
                # there can be some cases like fair_bike_02_3 where LLM gave a clarification, just focus on answer number and not the count
                # Use a dictionary to handle duplicate answer numbers
                answer_dict = {}
                for a_num, a_rewrite, a_text in answers:
                    answer_dict[a_num] = (a_rewrite, a_text)
                # Convert the dictionary back to a list and sort by answer number
                answers = sorted((int(num), rewrite, text) for num, (rewrite, text) in answer_dict.items())
                # Check if the number of questions matches the number of answers
                if len(questions) != len(answers):
                    
                    print(questions)
                    print("$"*100)
                    print(answers)

                    # if len(questions) > len(answers):
                        # questions = questions[:len(answers)]
                    # else:
                    raise ValueError(f"The number of questions does not match the number of answers. {len(questions)} vs {len(answers)}")

                # raise ValueError(f"The number of questions does not match the number of answers. {len(questions)} vs {len(answers)}")
                # print(questions)
                # print("$"*100)
                # print(answers)
                # exit()

            # Combine questions and answers
            qa_pairs = []
            # print("F"*100)
            # print(questions, answers)
            for (q_num, q_text), (a_num, a_rewrite, a_text) in zip(questions, answers):
                # print(q_num, a_num)
                if int(q_num) != int(a_num):
                    raise ValueError(f"Question number does not match answer number. {q_num} vs {a_num}")
                # assert q_text == a_rewrite, f"Two questions do not match: @{q_text}@ and @{a_rewrite}@"
                qa_pairs.append({'question': q_text, 'rewrite': a_rewrite, 'answer': a_text})

            parsed_data += qa_pairs

        return parsed_data

    except Exception as e:
        raise ValueError(f"Failed to parse the file: {str(e)}")

# Example usage
file_path = "/path/to/outputs/upenn_0714_Cooking_1_2.txt"

from tqdm import tqdm
import sys
missing_answers = {}

try:
    slurm_idx = int(sys.argv[1])
    print(f"Doing idx: {slurm_idx}...")
except:
    assert False, "Not supported now..."

for idx, file in enumerate(tqdm(os.listdir('outputs/'))):
    if idx % 50 != slurm_idx:
        continue
    if file in ["fair_bike_07_14.txt", "utokyo_pcr_2001_34_6.txt", "georgiatech_bike_09_11.txt", "sfu_cooking023_4.txt", "iiith_cooking_06_1.txt", "iiith_cooking_142_4.txt", "minnesota_cooking_010_2.txt", "indiana_bike_13_3.txt", "georgiatech_covid_07_6.txt", "indiana_cooking_23_4.txt", "indiana_cooking_26_3.txt", "indiana_cooking_27_2.txt", "nus_cpr_51_1.txt"]:
        continue
    print(file)
    result = parse_file(f"outputs/{file}")
    # print(len(result))
    # print(result[0].keys())

    # Find all object labels
    take_name = file[:-4]
    split_files = [x for x in os.listdir(detic_basedir) if take_name in x]
    all_objects = []
    for split_idx in range(len(split_files)):
        data = pd.read_csv(os.path.join(detic_basedir, split_files[split_idx]))
        all_objects += list(data['object_name'].unique())
    all_objects = list(set(all_objects))

    # Assert for question timestamps


    current_take_narrations = {}
    object_interaction_dataset = pd.DataFrame(columns=['timestamp', 'narration', 'object'])
    assert take2uid[take_name] in atomic_descriptions, f"{take_name} not found..."
    # print(len(atomic_descriptions[take2uid[take_name]]))
    for narration_pass_idx in range(len(atomic_descriptions[take2uid[take_name]])):
        if not atomic_descriptions[take2uid[take_name]][narration_pass_idx]['rejected']:
            for narration_text_idx in range(len(atomic_descriptions[take2uid[take_name]][narration_pass_idx]['descriptions'])):
                narr_d = atomic_descriptions[take2uid[take_name]][narration_pass_idx]['descriptions'][narration_text_idx]
                current_take_narrations[narr_d['text'].strip().replace("2.5", "25")] = 1
                row_to_append = pd.DataFrame([{'timestamp': float(narr_d['timestamp']), 'narration': narr_d['text'].strip().replace("2.5", "25"), 'object': -1}])
                object_interaction_dataset = pd.concat([object_interaction_dataset, row_to_append])
    print(current_take_narrations)
    print("$"*100)


    missing_answers[take_name] = []

    compiled_answers = {} # tuple of question and parsed answer, then map to the narration 
    for idx in range(len(result)):
        assert result[idx]['question'].strip() in current_take_narrations, f"Why is this not available?@{result[idx]['question']}@"
        final_answer = result[idx]['answer']
        current_objects = []
        current_valid_objects = []
        for candidate_object in all_objects:
            if candidate_object in final_answer:
                current_objects.append(candidate_object)
                current_valid_objects.append(candidate_object)
        if 'NO INTERACTION' in final_answer:
            current_objects.append('NO INTERACTION')
        if 'NO MATCHING OBJECTS' in final_answer:
            current_objects.append('NO MATCHING OBJECTS')
        if len(current_objects) == 0:
            print(f"Why is this zero?? ${final_answer}$")
            missing_answers[take_name].append(final_answer)
        if len(current_valid_objects) > 0:
            assert (object_interaction_dataset['narration'] == result[idx]['question'].strip()).any(), f"The following narration was not found in the narration set : {result[idx]['question'].strip()}"
            object_interaction_dataset.loc[object_interaction_dataset['narration'] == result[idx]['question'].strip(), 'object'] = "@@@@@".join(current_valid_objects)
            compiled_answers[result[idx]['question'].strip()] = current_valid_objects
        # assert len(current_objects) > 0, f"Why is this zero?? ${final_answer}$"
        # print(final_answer)
        # exit()
        # Find objects being mentioned, or placeholders. raise error if nothing found to investigate
        # if result[idx]['question'] != result[idx]['rewrite']: 
            # sim, score = are_similar(result[idx]['question'], result[idx]['rewrite'], threshold=0.50)
            # if not sim:
                # f2.write(f"Inconsistent: @{result[idx]['question']}@ vs @{result[idx]['rewrite']}@ Score: {score}\n")
                # print(f"Inconsistent: @{result[idx]['question']}@ vs @{result[idx]['rewrite']}@ Score: {score}")
                # assert False

        # print(result[idx]['question'], result[idx]['rewrite'])
    # exit()

    # Print the interaction after removing no objects or interactions
    object_interaction_dataset = object_interaction_dataset[~object_interaction_dataset['object'].apply(lambda x: isinstance(x, int))]
    print(object_interaction_dataset.head())
    object_interaction_dataset.to_csv(f'compiled_interactions_v1/{take_name}.csv', index=False)
    print("$"*100)

# with open(f'compiled_interactions/{slurm_idx}.json', 'w') as f:
    # json.dump(missing_answers, f)

exit()
num_success = 0
num_fails = 0
try:
    result = parse_file(file_path)
    print(result)
except ValueError as e:
    print(e)