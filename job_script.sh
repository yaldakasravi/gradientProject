#!/bin/bash -l
#SBATCH --account=def-ahafid # Use 'rrg' account for better chance to run GPU
#SBATCH --time=10:00:00  # Time you think your experiment will take. Experiment gets killed if this time is exceeded. Shorter experiments usually get priority in queue.
#SBATCH --ntasks=1 # Number of tasks per node. Generally keep as 1.
#SBATCH --cpus-per-task=4           # CPU cores/threads. 3.5 cores / gpu is standard.
#SBATCH --gres=gpu:a100:1                  # Number of GPUs (per node)
#SBATCH --mem=40Gb                  # RAM allowed per node
#SBATCH --output=%j_new_ml.out
# Load your required modules (e.g., Python, TensorFlow)
source ghostfacenets_env/bin/activate
module load StdEnv/2020
module load python/3.10.2
module load httpproxy
pip install --no-index tensorflow==2.8
pip install --no-index comet_ml
# Run your script

#python lfw.py
#python comet_perturb.py
#python comet_perturb-oneSide.py
#python masking.py
#python masking-oneSide.py
#python noise-masking.py
#python noise-masking-oneSide.py
#python random-masking.py
#python random-masking-oneSide.py
#python random-noise.py
#python random-noise-oneSide.py
#python swap.py
#python swap-sameperson.py
#python swap-oneSide.py
#python swap-sameperson-oneSide.py
#python swap-rand-differentperson.py
#python swap-rand-sameperson.py
#python swap-rand-differentperson-oneside.py
#python swap-rand-sameperson-oneside.py
