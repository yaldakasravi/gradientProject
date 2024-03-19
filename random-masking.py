import numpy as np
import os
from comet_ml import Experiment
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import tensorflow as tf

# Initialize your Comet ML experiment here
experiment = Experiment(api_key="UuHTEgYku8q9Ww3n13pSEgC8d", project_name="random-square-masking_effect", workspace="enhancing-gradient")

# Define paths
model_path = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/models/GN_W0.5_S2_ArcFace_epoch16.h5'
dataset_dir = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files_base = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files = [f'pairs_{i:02}.txt' for i in range(1, 11)]  # Adjust the range as needed

import random

def apply_random_mask(img_array, num_squares, square_size):
    """
    Applies a given number of square masks randomly on the image.

    :param img_array: NumPy array of the image.
    :param num_squares: The number of squares to apply.
    :param square_size: The size of each square.
    """
    h, w, _ = img_array.shape
    for _ in range(num_squares):
        x1 = random.randint(0, w - square_size)
        y1 = random.randint(0, h - square_size)
        img_array[y1:y1+square_size, x1:x1+square_size, :] = 0
    return img_array

def load_and_preprocess_image(image_path, num_squares=0, square_size=20):
    img = image.load_img(image_path, target_size=(112, 112))
    img_array = image.img_to_array(img)

    # Apply random masking if specified
    if num_squares > 0:
        img_array = apply_random_mask(img_array, num_squares, square_size)

    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)

def get_embedding(model, image_path, num_squares=0, square_size=20):
    processed_image = load_and_preprocess_image(image_path, num_squares=num_squares, square_size=square_size)
    return model.predict(processed_image)

def get_cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1.flatten(), embedding2.flatten())

def evaluate_lfw(model, dataset_dir, pairs_file_path, num_squares=0, square_size=20):
    similarities, labels = [], []
    with open(pairs_file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        for i in range(0, len(lines), 4):
            # First pair (same person)
            img1_path_same = os.path.join(dataset_dir, lines[i])
            img2_path_same = os.path.join(dataset_dir, lines[i+1])
            embedding1_same = get_embedding(model, img1_path_same, num_squares=num_squares, square_size=square_size)
            embedding2_same = get_embedding(model, img2_path_same, num_squares=num_squares, square_size=square_size)
            similarity_same = get_cosine_similarity(embedding1_same, embedding2_same)
            similarities.append(similarity_same)
            print(f"Similarity scores sample for {pairs_file_path}: {similarities[:10]}")

            print(f"Using threshold: {threshold}")

            labels.append(1)  # Same person

            # Second pair (different people)
            img1_path_diff = os.path.join(dataset_dir, lines[i+2])
            img2_path_diff = os.path.join(dataset_dir, lines[i+3])
            embedding1_diff = get_embedding(model, img1_path_diff, num_squares=num_squares, square_size=square_size)
            embedding2_diff = get_embedding(model, img2_path_diff, num_squares=num_squares, square_size=square_size)
            similarity_diff = get_cosine_similarity(embedding1_diff, embedding2_diff)
            similarities.append(similarity_diff)
            print(f"Similarity scores sample for {pairs_file_path}: {similarities[:10]}")

            print(f"Using threshold: {threshold}")
            labels.append(0)  # Different people

    return np.array(labels), np.array(similarities)

def calculate_metrics(labels, similarities, threshold):
    predictions = similarities >= threshold
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1

def main():
    model = load_model(model_path)
    metrics_summary = {}

    # Define ranges for number of squares and square size
    num_squares_range = range(1, 11)  # example: 1 to 10 squares
    square_size = 20  # example: each square is 20x20 pixels

    for num_squares in num_squares_range:
        metrics_results = [] 
        for pairs_file in pairs_files:
            pairs_file_path = os.path.join(pairs_files_base, pairs_file)
            labels, similarities = evaluate_lfw(model, dataset_dir, pairs_file_path, num_squares=num_squares, square_size=square_size)
            threshold = np.percentile(similarities, 65)
            accuracy, precision, recall, f1 = calculate_metrics(labels, similarities, threshold)
            metrics_results.append({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1})
            # Print summary for the current pairs file
            print(f"Summary for {pairs_file} with {num_squares} masks:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}\n")

        avg_metrics = {metric: np.mean([res[metric] for res in metrics_results]) for metric in ['accuracy', 'precision', 'recall', 'f1']}
        metrics_summary[num_squares] = avg_metrics
        # Log results to Comet ML
        experiment.log_metrics(avg_metrics, prefix=f"masks_{num_squares}")

    save_directory = "random-square-masking_plot"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Plot results across different levels of masking
    for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
        plt.figure()
        x = list(metrics_summary.keys())
        y = [metrics_summary[num_squares][metric_name] for num_squares in x]
        plt.plot(x, y, marker='o', linestyle='-')
        plt.title(f"{metric_name.capitalize()} vs. Number of Masks")
        plt.xlabel("Number of Masks")
        plt.ylabel(metric_name.capitalize())
        plt.grid(True)
        plt.savefig(os.path.join(save_directory, f"{metric_name}_across_pairs_files.png"))
        plt.close()

    experiment.end()

if __name__ == "__main__":
    main()

