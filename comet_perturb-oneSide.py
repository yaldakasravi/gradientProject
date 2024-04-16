import numpy as np
import os
from comet_ml import Experiment
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing import image
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input
#from PIL import Image
# Assuming the Comet ML experiment is correctly initialized
experiment = Experiment(api_key="UuHTEgYku8q9Ww3n13pSEgC8d", project_name="inoise-threshold-analysis", workspace="enhancing-gradient")

# Define paths
model_path = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/models/GN_W0.5_S2_ArcFace_epoch16.h5'
dataset_dir = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files_base = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files = [f'pairs_{i:02}.txt' for i in range(1, 11)]  # Adjust the range as needed
"""
def preprocess_image(image_path, noise_factor):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [112, 112])
    img = tf.cast(img, tf.float32)

    # Apply noise
    noise = tf.random.uniform(tf.shape(img), 0, 255)
    img = (1 - noise_factor) * img + noise_factor * noise
    img = tf.clip_by_value(img, 0, 255)

    return preprocess_input(img)

def load_and_preprocess_image(image_path, noise_factor):
    img = preprocess_image(image_path, noise_factor)
    return tf.expand_dims(img, 0)

def create_pairs_dataset(pairs_file_path, dataset_dir, noise_factor):
    def parse_line(line):
        parts = tf.strings.split(line)
        return parts[0], parts[1], tf.strings.to_number(parts[2], tf.int32)

    lines = tf.data.TextLineDataset(pairs_file_path).map(parse_line)
    
    def load_and_process_images(path1, path2, label):
        # Apply noise to the first image
        img1 = load_and_preprocess_image(tf.strings.join([dataset_dir, path1]), noise_factor)
        # Do not apply noise to the second image
        img2 = load_and_preprocess_image(tf.strings.join([dataset_dir, path2]), 0)  # Note the noise_factor is 0
        return (img1, img2), label

    return lines.map(load_and_process_images)

def get_embedding(model, processed_image):
    return model.predict(processed_image)

def calculate_cosine_similarity(embeddings):
    # Assuming embeddings is a list where each element is (embedding1, embedding2)
    similarities = [1 - cosine(e1.flatten(), e2.flatten()) for e1, e2 in embeddings]
    return np.array(similarities)

def calculate_metrics(labels, similarities, threshold):
    predictions = similarities >= threshold
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        'accuracy': accuracy,
    }

def main():
    with tf.device('/GPU:1'):  # Adjust GPU device as necessary
        model = load_model(model_path)
        noise_levels = np.linspace(0, 0.7, num=8)
        thresholds = np.linspace(0.3, 1, num=14)

        # Dictionary to hold the average accuracy for each noise level across thresholds
        avg_accuracy_per_noise_level = {noise: [] for noise in noise_levels}

        for noise_factor in noise_levels:
            print(f"Evaluating noise level: {noise_factor}")
            all_metrics = []

            for th in thresholds:
                # This part of the code assumes you have a function to evaluate the LFW dataset
                # which returns the labels and similarities for the given noise factor and threshold.
                # It should be adapted to use your actual evaluation function.
                # Example: metrics = evaluate_lfw_with_noise(model, dataset_dir, pairs_files, noise_factor, th)
                metrics = {'accuracy': np.random.rand()}  # Placeholder for actual metrics calculation
                all_metrics.append(metrics)

                avg_accuracy_per_noise_level[noise_factor].append(metrics['accuracy'])

        # Plotting
        save_directory = "noise_level_experiment_for_one_side_plots"
        os.makedirs(save_directory, exist_ok=True)
        # Define a color map for consistent colors across noise levels
        color_map = plt.cm.get_cmap('viridis', len(noise_levels))
        
        # Create the figure and axis objects
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(1, 1, 1)
        
        # Plot each noise level with its color from the color map
        for i, (noise_factor, accuracies) in enumerate(avg_accuracy_per_noise_level.items()):
            ax.plot(thresholds, accuracies, marker='o', linestyle='-', label=f'Noise {noise_factor:.2f}',
                    color=color_map(i), linewidth=2, markersize=6)
        
        # Enhance the plot
        ax.set_title("Accuracy vs. Threshold for different Noise Levels for one side")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Accuracy")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)  # Place the legend outside the plot
        ax.grid(True)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, "accuracy_vs_threshold_for_noise_levels_improved.png"))
        plt.close()
if __name__ == "__main__":
    main()
"""
#modification for the plotting 
def preprocess_image(file_path, noise_factor, apply_noise=False):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [112, 112])
    img = tf.cast(img, tf.float32) / 255.0

    # Optionally apply noise
    if apply_noise:
        noise = tf.random.uniform(tf.shape(img), 0, 1)
        img = (1 - noise_factor) * img + noise_factor * noise
    return img

def load_and_preprocess_image(path1, path2, noise_factor):
    img1 = preprocess_image(path1, noise_factor, apply_noise=True)
    img2 = preprocess_image(path2, 0, apply_noise=False)  # No noise applied
    return (img1, img2)

def parse_line(line, dataset_dir):
    parts = tf.strings.split(line)
    path1 = tf.strings.join([dataset_dir, parts[1]])
    path2 = tf.strings.join([dataset_dir, parts[3]])
    label = tf.strings.to_number(parts[4], tf.int32)
    return path1, path2, label

def prepare_dataset(pairs_file, dataset_dir, noise_factor, batch_size):
    lines = tf.data.TextLineDataset(pairs_file).skip(1)  # Skip the header line
    dataset = lines.map(lambda line: parse_line(line, dataset_dir))
    dataset = dataset.map(lambda path1, path2, label: ((load_and_preprocess_image(path1, path2, noise_factor)), label))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def calculate_cosine_similarity(features):
    f1, f2 = features[:, 0], features[:, 1]
    norm_f1 = tf.nn.l2_normalize(f1, axis=1)
    norm_f2 = tf.nn.l2_normalize(f2, axis=1)
    cos_sim = tf.reduce_sum(tf.multiply(norm_f1, norm_f2), axis=1)
    return cos_sim

def evaluate_model(model, dataset):
    labels = []
    predictions = []

    for (img1, img2), label in dataset:
        # Get embeddings
        embeddings = model(tf.concat([img1, img2], axis=0))
        f1, f2 = embeddings[:len(embeddings)//2], embeddings[len(embeddings)//2:]

        # Calculate cosine similarity
        cos_sim = calculate_cosine_similarity(tf.stack([f1, f2], axis=1))

        # Define your threshold for deciding if images match
        threshold = 0.5
        pred = tf.cast(cos_sim > threshold, tf.int32)

        labels.extend(label.numpy())
        predictions.extend(pred.numpy())

    return accuracy_score(labels, predictions)

def main():
    model = load_model(model_path)
    noise_levels = np.linspace(0, 1.0, 11)
    accuracies = {noise: [] for noise in noise_levels}

    # Evaluate each pairs file at each noise level
    for pairs_file in pairs_files:
        for noise_level in noise_levels:
            dataset = prepare_dataset(pairs_file, dataset_dir, noise_level, batch_size=32)
            accuracy = evaluate_model(model, dataset)
            accuracies[noise_level].append(accuracy)

    # Plotting results
    plt.figure(figsize=(10, 8))
    for noise_level, accs in accuracies.items():
        plt.plot(accs, label=f'Noise {noise_level:.2f}', marker='o')

    plt.title("Accuracy by Noise Level Across Different Pairs Files")
    plt.xlabel("Pairs File Index")
    plt.ylabel("Accuracy")
    plt.xticks(range(len(pairs_files)), [f'{i+1}' for i in range(len(pairs_files))])
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting
    save_directory = "noise_level_experiment_plots_oneSide"
    os.makedirs(save_directory, exist_ok=True)

   # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "accuracy_vs_threshold_for_noise_levels_improved.png"))
    plt.close()

if __name__ == "__main__":
    main()
