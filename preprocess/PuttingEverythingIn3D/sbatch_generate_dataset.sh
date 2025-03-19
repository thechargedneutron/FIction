#!/bin/bash

#SBATCH --job-name=process_dataset
#SBATCH --partition=fill-here
#SBATCH --nodes=1
#SBATCH --array=0-1438
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node! 1437
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --gres=gpu:0                 # number of gpus
#SBATCH --time 48:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/op_%A_%a.out           # output file name
#SBATCH --error=logs/op_%A_%a.err
#SBATCH --constraint='volta32gb'
#SBATCH --mem=60G
#SBATCH --open-mode=append

#SBATCH --mail-user=your-email@gmail.com
#SBATCH --mail-type=begin,end,fail # mail once the job finishes



python main.py $SLURM_ARRAY_TASK_ID 