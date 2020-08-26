#!/bin/bash

#SBATCH --job-name=glb5
#SBATCH --output=JobInfo/%x_%a.out
#SBATCH --error=JobInfo/%x_%a.err
#SBATCH --array=0-999
#SBATCH --ntasks=1
#SBATCH -p konings,owners,normal
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2000

######################
# Begin work section #
######################

# Print this sub-job's task ID
echo "GRID: " $SLURM_ARRAY_TASK_ID >> Retrieval_5.out
python Retrieval_0817.py 5
python Retrieval_0817.py 17
python Retrieval_0817.py 29
python Retrieval_0817.py 41
python Retrieval_0817.py 53
python Retrieval_0817.py 65
python Retrieval_0817.py 77
python Retrieval_0817.py 89
