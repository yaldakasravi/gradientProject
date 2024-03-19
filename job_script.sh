#!/bin/bash -l
#SBATCH --account=def-ahafid # Use 'rrg' account for better chance to run GPU
#SBATCH --time=06:00:00  # Time you think your experiment will take. Experiment gets killed if this time is exceeded. Shorter experiments usually get priority in queue.
#SBATCH --ntasks=1 # Number of tasks per node. Generally keep as 1.
#SBATCH --cpus-per-task=4           # CPU cores/threads. 3.5 cores / gpu is standard.
#SBATCH --gres=gpu:a100:1                  # Number of GPUs (per node)
#SBATCH --mem=40Gb                  # RAM allowed per node
#SBATCH --output=%j_new_ml.out
# Load your required modules (e.g., Python, TensorFlow)
source ghostfacenets_env/bin/activate
module load python/3.10.2
module load httpproxy
pip install --no-index tensorflow==2.8
pip install --no-index comet_ml
# Run your script

python comet_perturb.py
#python masking.py
#python random-masking.py
#python swap.py
