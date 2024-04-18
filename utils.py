import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import random

# Parameters
model_path = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/models/GN_W0.5_S2_ArcFace_epoch16.h5'
dataset_dir = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
#pairs_files = [os.path.join(dataset_dir, f'pairs_{i:02}.txt') for i in range(1, 11)]
pairs_files = [os.path.join(dataset_dir, f'pairs_{i:02}.txt') for i in range(1, 2)]


def read_pairs(pairs_file):
    pairs = []
    with open(pairs_file, "r") as file:
        lines = file.readlines()
        stop_width = 5
        for i in range(0, int(len(lines)/stop_width) ):
                ind = i * stop_width
                file1 = os.path.join(dataset_dir, lines[ind].strip())
                file2 = os.path.join(dataset_dir, lines[ind + 1].strip())
                file3 = os.path.join(dataset_dir, lines[ind + 2].strip())
                file4 = os.path.join(dataset_dir, lines[ind + 3].strip())
                if os.path.isfile(file1) and os.path.isfile(file2):
                    pairs.append((file1, file2, True))
                if os.path.isfile(file3) and os.path.isfile(file4):
                    pairs.append((file3, file4, False))
    return pairs
