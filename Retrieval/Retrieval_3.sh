#!/bin/bash

#SBATCH --job-name=glb3
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
echo "GRID: " $SLURM_ARRAY_TASK_ID >> Retrieval_3.out
python Retrieval_0817.py 3
python Retrieval_0817.py 15
python Retrieval_0817.py 27
python Retrieval_0817.py 39
python Retrieval_0817.py 51
python Retrieval_0817.py 63
python Retrieval_0817.py 75
python Retrieval_0817.py 87
