#!/bin/bash

#SBATCH --job-name=train
#SBATCH --partition=partition
#SBATCH --nodes=1
#SBATCH --array=0-5
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node! 1437
#SBATCH --cpus-per-task=80           # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 48:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=/full/path/to/logs/train_%A_%a.out           # output file name
#SBATCH --error=/full/path/to/logs/train_%A_%a.err
#SBATCH --constraint='volta32gb'
#SBATCH --mem=480G
#SBATCH --open-mode=append

#SBATCH --mail-user=email@gmail.com
#SBATCH --mail-type=begin,end,fail # mail once the job finishes


# Define the arrays for hyperparameter search
learning_rates=(5e-5 1e-4)
future_times=(60 120 180)
observation_time=(60)
training_task="pose-only"

# Get the number of elements in each array
len_lr=${#learning_rates[@]}
len_ft=${#future_times[@]}
len_ot=${#observation_time[@]}

# Calculate the total number of combinations
total_combinations=$((len_lr * len_ft * len_ot))

# SLURM_ARRAY_TASK_ID is used to find the index of the current job
task_id=${SLURM_ARRAY_TASK_ID}

# Calculate the index for each array
index_lr=$((task_id / (len_ft * len_ot)))
index_ft=$(((task_id / len_ot) % len_ft))
index_ot=$((task_id % len_ot))

# Get the correct values based on the computed indices
learning_rate=${learning_rates[$index_lr]}
future_time=${future_times[$index_ft]}
observation_time=${observation_time[$index_ot]}

# Print the chosen parameters for debugging
echo "Running with learning_rate=$learning_rate and future_time=$future_time and observation_time=$observation_time"

expt_name="${SLURM_JOB_NAME}_${SLURM_JOB_ID}_lr_${learning_rate}_ft_${future_time}_obst_${observation_time}"

# Run your Python script with the chosen parameters
CUDA_VISIBLE_DEVICES=7,6,5,4,3,2,1,0 python main.py \
 --expt_name $expt_name \
 --training_task $training_task \
 --learning_rate $learning_rate \
 --future_time $future_time \
 --observation_time $observation_time \
 --use_video True \
 --use_pose True \
 --use_location True \
 --use_env True