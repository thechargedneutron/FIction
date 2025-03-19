#!/bin/bash

if [ "$1" = "" ]; then
    echo "Job name cannot be empty"
    exit 1
fi

export JOB_NAME=$(date '+%Y-%m-%d_%H:%M:%S_')$1
export DESTINATION_DIR='/path/to/checkpoints/'

mkdir $DESTINATION_DIR/$JOB_NAME/
cp loader.py $DESTINATION_DIR/$JOB_NAME/
cp main.py $DESTINATION_DIR/$JOB_NAME/
cp model.py $DESTINATION_DIR/$JOB_NAME/
cp pose_utils.py $DESTINATION_DIR/$JOB_NAME/
cp utils.py $DESTINATION_DIR/$JOB_NAME/
cp trainer.py $DESTINATION_DIR/$JOB_NAME/
cp sbatch_train_hp.sh $DESTINATION_DIR/$JOB_NAME/sbatch.sh

cd $DESTINATION_DIR/$JOB_NAME/

sbatch -J $JOB_NAME sbatch.sh
