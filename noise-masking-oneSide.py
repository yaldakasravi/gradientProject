import numpy as np
import os
from comet_ml import Experiment
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import tensorflow as tf
from tensorflow.data import Dataset

# Initialize your Comet ML experiment here
experiment = Experiment(api_key="UuHTEgYku8q9Ww3n13pSEgC8d", project_name="masking_effect", workspace="enhancing-gradient")

# Define paths
model_path = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/models/GN_W0.5_S2_ArcFace_epoch16.h5'
dataset_dir = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files_base = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files = [f'pairs_{i:02}.txt' for i in range(1, 11)]  # Adjust the range as needed

#very slow 
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
#make it noisy 
def main():
    with tf.device('/GPU:1'):
        model = load_model(model_path)
        thresholds = np.linspace(0.3, 1, num=14)
        noise_levels = np.linspace(0.0, 1.0, num=11)

        # Initialize dictionary to store average accuracy for each mask thickness level across thresholds
        avg_accuracy_per_thickness_level = {noise: [] for noise in noise_levels}

        for th in thresholds:
            avg_metrics = {noise: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for noise in noise_levels}
            
            for noise_factor in noise_levels:
                all_metrics = []

                for pairs_file in pairs_files:
                    pairs_file_path = os.path.join(pairs_files_base, pairs_file)
                    y_true, y_pred_scores = evaluate_lfw(model, dataset_dir, pairs_file_path, noise_factor=noise_factor)
                    metrics = calculate_metrics(y_true, y_pred_scores, th)
                    all_metrics.append(metrics)

                # Calculate and print average metrics for this mask thickness level
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    metric_values = [m[metric] for m in all_metrics]
                    avg_metrics[noise_factor][metric] = np.mean(metric_values)

                print(f"Summary for mask thickness {noise_factor} and threshold {th}:")
                print(f"  Accuracy: {avg_metrics[noise_factor]['accuracy']:.4f}")
                print(f"  Precision: {avg_metrics[noise_factor]['precision']:.4f}")
                print(f"  Recall: {avg_metrics[noise_factor]['recall']:.4f}")
                print(f"  F1 Score: {avg_metrics[noise_factor]['f1']:.4f}\n")

            # Store average accuracy for each mask thickness level for this threshold
            for noise_factor in noise_levels:
                avg_accuracy_per_thickness_level[noise_factor].append(avg_metrics[noise_factor]['accuracy'])

        save_directory = "threshold-mask-noise_plot"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Plotting for each metric could be done here, similar to the structure in your example
        # Example for accuracy plot across all thresholds
        plt.figure()
        for mask_thickness in mask_thickness_levels:
            plt.plot(thresholds, avg_accuracy_per_thickness_level[noise_factor], marker='o', linestyle='-', label=f'noise Thickness {noise_factor}')
        plt.title("Accuracy vs. Threshold for Different Mask Thicknesses")
        plt.xlabel("Threshold")
        plt.ylabel("Accuracy")
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(os.path.join(save_directory, "accuracy_vs_threshold_for_mask_thicknesses.png"))
        plt.close()

if __name__ == "__main__":
    main()
"""

#faster with dataloader 

def preprocess_image(image_path, noise_factor):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [112, 112])
    img = tf.cast(img, tf.float32) / 255.0  # Normalize the image to [0, 1]

    # Add Gaussian noise
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=noise_factor, dtype=tf.float32)
    img = img + noise
    img = tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)

    # Preprocess the image for the model
    img = preprocess_input(img)
    return img

def get_embedding(model, processed_image):
    return model.predict(tf.expand_dims(processed_image, axis=0))

def create_pairs_dataset(pairs_file_path, dataset_dir, noise_factor):
    def parse_function(line):
        parts = tf.strings.split(line)
        # Ensure file paths are joined with dataset_dir properly
        file1_path = tf.strings.join([dataset_dir, '/', parts[0]], separator='')
        file2_path = tf.strings.join([dataset_dir, '/', parts[1]], separator='')
        label = tf.strings.to_number(parts[2], tf.int32)
        return file1_path, file2_path, label

    lines_dataset = tf.data.TextLineDataset(pairs_file_path).map(parse_function)

    # Debugging: Print out the structure of dataset elements
    for element in lines_dataset.take(1):
        print(element)

    def load_and_preprocess(pair):
        # Unpack the tuple
        file_path1, file_path2, label = pair
        img1 = preprocess_image(file_path1, noise_factor)
        img2 = preprocess_image(file_path2, 0)
        return (img1, img2), label

    # Use a lambda to ensure the function takes a single argument
    pairs_dataset = lines_dataset.map(lambda pair: load_and_preprocess((pair[0], pair[1], pair[2])))
    return pairs_dataset

def compute_similarity(embedding1, embedding2):
    # Cosine similarity function
    dot_product = tf.reduce_sum(embedding1 * embedding2, axis=1)
    norm_product = tf.norm(embedding1, axis=1) * tf.norm(embedding2, axis=1)
    return 1 - dot_product / norm_product

# Adjust the main function
def main():
    with tf.device('/GPU:0'):
        model = load_model(model_path)
        thresholds = np.linspace(0.3, 1, num=14)
        noise_levels = np.linspace(0.0, 1.0, num=11)

        avg_accuracy_per_noise_level = {noise: [] for noise in noise_levels}

        for noise_factor in noise_levels:
            metrics_per_threshold = {th: [] for th in thresholds}

            for pairs_file in pairs_files:
                pairs_file_path = os.path.join(pairs_files_base, pairs_file)
                pairs_dataset = create_pairs_dataset(pairs_file_path, dataset_dir, noise_factor).batch(32)

                for images, labels in pairs_dataset:
                    img1, img2 = images
                    embeddings1 = get_embedding(model, img1)
                    embeddings2 = get_embedding(model, img2)
                    similarities = compute_similarity(embeddings1, embeddings2).numpy()

                    for th in thresholds:
                        predictions = similarities >= th
                        accuracy = np.mean(predictions == labels.numpy())
                        metrics_per_threshold[th].append(accuracy)

            # Compute the average metrics for this noise level
            for th, accuracies in metrics_per_threshold.items():
                avg_accuracy = np.mean(accuracies)
                avg_accuracy_per_noise_level[noise_factor].append(avg_accuracy)

            print(f"Summary for noise level {noise_factor}:")
            for th in thresholds:
                print(f"  Threshold {th}: Accuracy: {avg_accuracy_per_noise_level[noise_factor][th]:.4f}")

        # Plotting
        save_directory = "threshold-mask-noise-oneSide_plot"
        os.makedirs(save_directory, exist_ok=True)

        for noise_factor, accuracies in avg_accuracy_per_noise_level.items():
            plt.figure()
            plt.plot(thresholds, accuracies, marker='o', linestyle='-', label=f'Noise {noise_factor}')
            plt.title(f"Accuracy vs. Threshold for Noise {noise_factor}-oneSide")
            plt.xlabel("Threshold")
            plt.ylabel("Accuracy")
            plt.legend(loc='best')
            plt.grid(True)
            plt.savefig(os.path.join(save_directory, f"accuracy_vs_threshold_noise_{noise_factor}.png"))
            plt.close()

if __name__ == "__main__":
    main()
