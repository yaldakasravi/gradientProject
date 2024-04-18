from utils import read_pairs
from comet_ml import Experiment
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import random

# Initialize your Comet ML experiment
experiment = Experiment(api_key="UuHTEgYku8q9Ww3n13pSEgC8d", project_name="swap-same-person-one-side-randsquare_plot", workspace="enhancing-gradient")

# Parameters
model_path = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/models/GN_W0.5_S2_ArcFace_epoch16.h5'
dataset_dir = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
#pairs_files = [os.path.join(dataset_dir, f'pairs_{i:02}.txt') for i in range(1, 11)]
pairs_files = [os.path.join(dataset_dir, f'pairs_{i:02}.txt') for i in range(1, 2)]
thresholds = np.linspace(0.0, 1, num=20)
num_squares_range = range(1, 11)
square_size = 20

# Load the model
model = load_model(model_path)

def preprocess_image(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Expected file but got directory or non-existent path: {image_path}")
    image = Image.open(image_path).resize((112, 112))
    image = np.array(image, dtype='float32')
    image /= 255.0  # Normalize
    return image

def swap_random_squares(image1, image2, num_squares, square_size):
    h, w, _ = image1.shape
    for _ in range(num_squares):
        x = random.randint(0, w - square_size)
        y = random.randint(0, h - square_size)
        # Swap the square from image2 to image1
        image1[y:y+square_size, x:x+square_size] = image2[y:y+square_size, x:x+square_size]
    return image1

def get_another_image_path(current_image_path):
    person_dir = os.path.dirname(current_image_path)
    all_images = [os.path.join(person_dir, img) for img in os.listdir(person_dir)
                  if os.path.join(person_dir, img) != current_image_path]
    if not all_images:
        return None
    return random.choice(all_images)

def calculate_similarity(image1, image2):
    emb1 = model.predict(np.expand_dims(image1, axis=0))
    emb2 = model.predict(np.expand_dims(image2, axis=0))
    similarity = np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity


def main():
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

                    # Get another image from the same person and swap squares
                    another_image_path = get_another_image_path(file1)
                    if another_image_path:
                        another_image = preprocess_image(another_image_path)
                        image1 = swap_random_squares(image1, another_image, num_squares, square_size)

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
                    accuracies.append(0)
                else:
                    accuracies.append((tp + tn) / total_comparisons)
            results[num_squares][threshold] = np.mean(accuracies)

    # Plotting
    save_directory = "one-swap-same-person-one-side-randsquare_plot"
    os.makedirs(save_directory, exist_ok=True)

    plt.figure(figsize=(10, 8))
    for num_squares, accuracies_by_threshold in results.items():
        thresholds_list = list(accuracies_by_threshold.keys())
        accuracies_list = [accuracies_by_threshold[th] for th in thresholds_list]
        plt.plot(thresholds_list, accuracies_list, label=f'Num Squares {num_squares}')

    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Effect of Swapping Random Squares with Same Person on Image1')
    plt.legend(title='Number of Squares')
    plt.grid(True)
    save_path = os.path.join(save_directory, 'accuracy_vs_thresholds_by_num_squares.png')
    plt.savefig(save_path)
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()

