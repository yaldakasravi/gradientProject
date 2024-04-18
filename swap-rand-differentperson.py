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

# TensorFlow compatible image loading and preprocessing

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

def swap_random_squares(image1, image2, num_squares, square_size):
    h, w, _ = image1.shape
    for _ in range(num_squares):
        x = random.randint(0, w - square_size)
        y = random.randint(0, h - square_size)
        temp = image1[y:y+square_size, x:x+square_size].copy()
        image1[y:y+square_size, x:x+square_size] = image2[y:y+square_size, x:x+square_size]
        image2[y:y+square_size, x:x+square_size] = temp
    return image1, image2

def get_different_person_image(current_image_path, dataset_dir):
    current_person = os.path.basename(os.path.dirname(current_image_path))
    all_people = [person for person in os.listdir(dataset_dir) if person != current_person and os.path.isdir(os.path.join(dataset_dir, person))]
    if not all_people:
        return None
    different_person = random.choice(all_people)
    different_person_dir = os.path.join(dataset_dir, different_person)
    different_person_images = os.listdir(different_person_dir)
    if not different_person_images:
        return None
    return os.path.join(different_person_dir, random.choice(different_person_images))
    
def get_another_image_path(same_person_dir, exclude_path):
    # Get all images except the one to exclude
    all_images = [os.path.join(same_person_dir, img) for img in os.listdir(same_person_dir)
                  if os.path.join(same_person_dir, img) != exclude_path]
    # Exclude directories and the image itself
    all_images = [img for img in all_images if os.path.isfile(img)]
    return random.choice(all_images) if len(all_images) > 1 else None

def calculate_similarity(image1, image2):
    emb1 = model.predict(np.expand_dims(image1, axis=0))
    emb2 = model.predict(np.expand_dims(image2, axis=0))
    similarity = np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity


def main():
    num_squares_range = range(1, 11)  # Example: 1 to 10 squares
    square_size = 20  # Square size
    results = {num_squares: {threshold: [] for threshold in thresholds} for num_squares in num_squares_range}

    for num_squares in num_squares_range:
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
                    different_person_image_path = get_different_person_image(file1, dataset_dir)
                    if different_person_image_path:
                        different_person_image = preprocess_image(different_person_image_path)
                        image1, image2 = swap_random_squares(image1, different_person_image, num_squares, square_size)

                    # Calculate similarity and update accuracy metrics
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
            results[num_squares][threshold] = np.mean(accuracies)

    # Plotting
    save_directory = "one-swap-different-person-both-sides-rand-square_plot"
    os.makedirs(save_directory, exist_ok=True)

    plt.figure(figsize=(10, 8))
    for level, accuracies_by_threshold in results.items():
        thresholds_list = list(accuracies_by_threshold.keys())
        accuracies_list = [accuracies_by_threshold[th] for th in thresholds_list]
        plt.plot(thresholds_list, accuracies_list, label=f'Mask Level {level:.2f}')

    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Effect of swapping random square with different person on Both Images')
    plt.legend(title='Mask Level')
    plt.grid(True)
    save_path = os.path.join(save_directory, 'accuracy_vs_thresholds_by_mask_level.png')
    plt.savefig(save_path)
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
