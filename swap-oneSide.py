from comet_ml import Experiment
from utils import read_pairs
# Initialize your Comet ML experiment here
experiment = Experiment(api_key="UuHTEgYku8q9Ww3n13pSEgC8d", project_name="masking_effect", workspace="enhancing-gradient")

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
thresholds = np.linspace(0.0, 1, num=20)
mask_thickness_levels = np.linspace(0, 1, num=10)

# Load the model
model = load_model(model_path)

def preprocess_image(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Expected file but got directory or non-existent path: {image_path}")
    image = Image.open(image_path).resize((112, 112))
    image = np.array(image, dtype='float32')
    image /= 255.0  # Normalize
    return image

def get_random_image_path(exclude_paths):
    all_images = []
    # Assuming dataset_dir is a directory containing subdirectories for each person
    for person_dir in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_dir)
        if os.path.isdir(person_path):
            all_images.extend([
                os.path.join(person_path, image)
                for image in os.listdir(person_path)
                if os.path.join(person_path, image) not in exclude_paths
            ])
    return random.choice(all_images) if all_images else None

def swap_eyes_area(image1, image2, eye_position, eye_size):
    x, y = eye_position
    eye_width, eye_height = eye_size

    # Assume eye regions are horizontal and have the same size
    swap_region1 = image1[y:y+eye_height, x:x+eye_width, :]
    swap_region2 = image2[y:y+eye_height, x:x+eye_width, :]
    image1[y:y+eye_height, x:x+eye_width, :] = swap_region2
    image2[y:y+eye_height, x:x+eye_width, :] = swap_region1

    return image1, image2

def calculate_similarity(image1, image2):
    emb1 = model.predict(np.expand_dims(image1, axis=0))
    emb2 = model.predict(np.expand_dims(image2, axis=0))
    similarity = np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity


def main():
    # Define eye position and size
    eye_position = (34, 56)  # Rough position for the eye region
    eye_size = (20, 10)  # Size of the eye region (width, height)

    results = {level: {threshold: [] for threshold in thresholds} for level in mask_thickness_levels}
    for level in mask_thickness_levels:
        for threshold in thresholds:
            accuracies = []
            for pairs_file in pairs_files:
                pairs = read_pairs(pairs_file)
                if not pairs:
                    continue
                tp = fp = tn = fn = 0
                for file1, file2, is_same in pairs:
                    image1 = preprocess_image(file1)
                    image2 = preprocess_image(file2)
                    # Select a random image to swap eyes from
                    random_image_path = get_random_image_path([file1, file2])
                    if random_image_path:
                        random_image = preprocess_image(random_image_path)

                        # Swap eye regions between image1, image2, and a random image
                        image1, _ = swap_eyes_area(image1, random_image, eye_position, eye_size)
                        #image2, _ = swap_eyes_area(image2, random_image, eye_position, eye_size)

                    similarity = calculate_similarity(image1, image2)
                    is_positive_match = similarity > threshold
                    if is_positive_match and is_same:
                        tp += 1
                    elif is_positive_match and not is_same:
                        fp += 1
                    elif not is_positive_match and not is_same:
                        tn += 1
                    elif not is_positive_match and is_same:
                        fn += 1
                total_comparisons = tp + fp + tn + fn
                if total_comparisons == 0:
                    accuracy = 0
                else:
                    accuracy = (tp + tn) / total_comparisons
                accuracies.append(accuracy)
            results[level][threshold] = np.mean(accuracies)

    # Plotting
    save_directory = "one-swap-different-person-one-sides_plot"
    os.makedirs(save_directory, exist_ok=True)

    plt.figure(figsize=(10, 8))
    for level, accuracies_by_threshold in results.items():
        thresholds_list = list(accuracies_by_threshold.keys())
        accuracies_list = [accuracies_by_threshold[th] for th in thresholds_list]
        plt.plot(thresholds_list, accuracies_list, label=f'Mask Level {level:.2f}')

    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Effect of swapping eyes with different person on one Images')
    plt.legend(title='Mask Level')
    plt.grid(True)
    save_path = os.path.join(save_directory, 'accuracy_vs_thresholds_by_mask_level.png')
    plt.savefig(save_path)
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()

