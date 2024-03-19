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
experiment = Experiment(api_key="UuHTEgYku8q9Ww3n13pSEgC8d", project_name="masking_effect", workspace="enhancing-gradient")

# Define paths
model_path = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/models/GN_W0.5_S2_ArcFace_epoch16.h5'
dataset_dir = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files_base = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files = [f'pairs_{i:02}.txt' for i in range(1, 11)]  # Adjust the range as needed

"""
#make it noisy 
def load_and_preprocess_image(image_path, noise_factor):
    img = image.load_img(image_path, target_size=(112, 112))  # Load the image with the target size
    img_array = image.img_to_array(img)  # Convert to a NumPy array

    # Define the region of interest where noise will be added
    mask_region = img_array[40:60, 30:-30, :]

    # Define the standard deviation of the Gaussian noise
    noise_sigma = noise_factor  # This controls how "noisy" the image will be

    # Generate Gaussian noise
    noise = np.random.normal(loc=0.0, scale=noise_sigma, size=mask_region.shape)

    # Since img_array is likely to be uint8, noise also needs to be in the same type
    noise = noise.astype(np.uint8)

    # Add the noise to the specified part of the image array
    img_array[40:60, 30:-30, :] = mask_region + noise

    # Ensure the values are still in the range [0, 255]
    img_array = np.clip(img_array, 0, 255)

    # Preprocess the image for the model
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_input(img_array_expanded_dims.astype(np.float32))

    return preprocess_input(img_array_expanded_dims)
"""
#make it black
def load_and_preprocess_image(image_path, mask_thickness):
    img = image.load_img(image_path, target_size=(112, 112))  # Load the image with the target size
    img_array = image.img_to_array(img)  # Convert to a NumPy array

    # Define the number of rows to mask based on the mask thickness
    num_rows_to_mask = int(112 * mask_thickness)  # convert thickness proportion to number of rows

    # Mask a horizontal strip across the center of the image
    start_row = (112 - num_rows_to_mask) // 2
    end_row = start_row + num_rows_to_mask
    img_array[start_row:end_row, :, :] = 0  # Set the pixel values to black

    # Preprocess the image for the model
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)


def get_embedding(model, image_path, noise_factor):
    processed_image = load_and_preprocess_image(image_path, noise_factor)
    return model.predict(processed_image)

def get_cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1.flatten(), embedding2.flatten())

def evaluate_lfw(model, dataset_dir, pairs_file_path, noise_factor):
    similarities, labels = [], []
    with open(pairs_file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        # Assuming each block of 4 lines in the pairs file corresponds to two pairs of images
        for i in range(0, len(lines), 4):
            # First pair (same person)
            img1_path_same = os.path.join(dataset_dir, lines[i])
            img2_path_same = os.path.join(dataset_dir, lines[i+1])
            #print("Loading same pair:", img1_path_same, img2_path_same)
            embedding1_same = get_embedding(model, img1_path_same, noise_factor)
            embedding2_same = get_embedding(model, img2_path_same, noise_factor)
            similarity_same = get_cosine_similarity(embedding1_same, embedding2_same)
            similarities.append(similarity_same)
            labels.append(1)  # Same person

            # Second pair (different people)
            img1_path_diff = os.path.join(dataset_dir, lines[i+2])
            img2_path_diff = os.path.join(dataset_dir, lines[i+3])
            embedding1_diff = get_embedding(model, img1_path_diff, noise_factor)
            embedding2_diff = get_embedding(model, img2_path_diff, noise_factor)
            similarity_diff = get_cosine_similarity(embedding1_diff, embedding2_diff)
            similarities.append(similarity_diff)
            labels.append(0)  # Different people

    return np.array(labels), np.array(similarities)

def calculate_metrics(labels, similarities,threshold):
    predictions = similarities >= threshold
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

#make it black 
def main():
    model = load_model(model_path)
    metrics_results = []
    fixed_threshold = 0.65
    # Instead of noise_levels, define mask thickness levels (e.g., 10% to 100% of the image height)
    mask_thickness_levels = np.linspace(0.1, 1.0, num=10)

    avg_metrics = {thickness: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for thickness in mask_thickness_levels}

    for mask_thickness in mask_thickness_levels:
        all_metrics = []
        for pairs_file in pairs_files:
            pairs_file_path = os.path.join(pairs_files_base, pairs_file)
            y_true, y_pred_scores = evaluate_lfw(model, dataset_dir, pairs_file_path, mask_thickness)
            metrics = calculate_metrics(y_true, y_pred_scores, fixed_threshold)
            all_metrics.append(metrics)
            result = calculate_metrics(y_true, y_pred_scores, fixed_threshold)
            if isinstance(result, dict):
                all_metrics.append(result)
                accuracy = result['accuracy']
                precision = result['precision']
                recall = result['recall']
                f1 = result['f1']
            else:
                print("Unexpected result type:", type(result))
            # Extract metrics from the returned dictionary
            #accuracy = metrics['accuracy']
            #precision = metrics['precision']
            #recall = metrics['recall']
            #f1 = metrics['f1']

            # Print summary for the current pairs file
            print(f"Summary for {pairs_file} with {mask_thickness} :")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}\n")

        # Calculate average metrics for this noise level
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            metric_values = [m[metric] for m in all_metrics]
            avg_metrics[mask_thickness][metric] = np.mean(metric_values)
    save_directory = "black-masking_plot"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
        plt.figure()
        thickness_factors = list(mask_thickness_levels)
        metric_values = [avg_metrics[thickness][metric_name] for thickness in thickness_factors]
        plt.plot(thickness_factors, metric_values, marker='o', linestyle='-')
        plt.title(f"{metric_name.capitalize()} vs. Mask Thickness")
        plt.xlabel("Mask Thickness")
        plt.ylabel(metric_name.capitalize())
        plt.grid(True)
        plt.savefig(os.path.join(save_directory, f"{metric_name}_vs_mask_thickness.png"))
        plt.close()
#make it noisy 
"""
def main():
    model = load_model(model_path)
    metrics_results = []
    # Fixed threshold value
    fixed_threshold = 0.65

    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    avg_metrics = {noise: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for noise in noise_levels}

    for noise_factor in noise_levels:
        all_metrics = []
        for pairs_file in pairs_files:
            pairs_file_path = os.path.join(pairs_files_base, pairs_file)
            y_true, y_pred_scores = evaluate_lfw(model, dataset_dir, pairs_file_path, noise_factor=noise_factor)
            metrics = calculate_metrics(y_true, y_pred_scores, fixed_threshold)  # ensure this returns a dict
            all_metrics.append(metrics)
            result = calculate_metrics(y_true, y_pred_scores, fixed_threshold)
            if isinstance(result, dict):
                all_metrics.append(result)
                accuracy = result['accuracy']
                precision = result['precision']
                recall = result['recall']
                f1 = result['f1']
            else:
                print("Unexpected result type:", type(result))
            # Extract metrics from the returned dictionary
            #accuracy = metrics['accuracy']
            #precision = metrics['precision']
            #recall = metrics['recall']
            #f1 = metrics['f1']

            # Print summary for the current pairs file
            print(f"Summary for {pairs_file} with {noise_factor} masks:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}\n")

        # Calculate average metrics for this noise level
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            metric_values = [m[metric] for m in all_metrics]
            avg_metrics[noise_factor][metric] = np.mean(metric_values)

    save_directory = "noise-masking_plot"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
        plt.figure()
        noise_factors = list(noise_levels)
        metric_values = [avg_metrics[noise][metric_name] for noise in noise_levels]
        plt.plot(noise_factors, metric_values, marker='o', linestyle='-')
        plt.title(f"{metric_name.capitalize()} vs. Noise Level")
        plt.xlabel("Noise Level")
        plt.ylabel(metric_name.capitalize())
        plt.grid(True)
        plt.savefig(os.path.join(save_directory, f"{metric_name}_vs_noise_level.png"))
        plt.close()
"""
if __name__ == "__main__":
    main()
