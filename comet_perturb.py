from utils import read_pairs
from comet_ml import Experiment
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
# Initialize your Comet ML experiment here
experiment = Experiment(api_key="UuHTEgYku8q9Ww3n13pSEgC8d", project_name="full-noise_effect", workspace="enhancing-gradient")

# Define paths
model_path = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/models/GN_W0.5_S2_ArcFace_epoch16.h5'
dataset_dir = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files_base = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
#pairs_files = [os.path.join(dataset_dir, f'pairs_{i:02}.txt') for i in range(1, 11)]
pairs_files = [os.path.join(dataset_dir, f'pairs_{i:02}.txt') for i in range(1, 2)]
# Load the model
model = load_model(model_path)

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
        img1 = load_and_preprocess_image(tf.strings.join([dataset_dir, path1]), noise_factor)
        img2 = load_and_preprocess_image(tf.strings.join([dataset_dir, path2]), noise_factor)
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
        save_directory = "noise_level_experiment_plots"
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
        ax.set_title("Accuracy vs. Threshold for different Noise Levels")
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

#make it faster 
def preprocess_image(image_path):
    image = Image.open(image_path).resize((112, 112))
    image = np.array(image, dtype='float32')
    image /= 255.0  # Normalize
    return image

def add_noise_to_image(image, level):
    # Calculate the noise intensity
    noise_intensity = level * 255  # Scale by 255 as pixel values range from 0 to 255
    
    # Generate noise for the entire image size
    noise = np.random.normal(loc=0.0, scale=noise_intensity, size=image.shape)

    # Add noise to the entire image
    noisy_image = image + noise
    
    # Ensure pixel values remain within [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)

    return noisy_image

def calculate_similarity(image1, image2):
    emb1 = model.predict(np.expand_dims(image1, axis=0))
    emb2 = model.predict(np.expand_dims(image2, axis=0))
    similarity = np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity


def main():
    thresholds = np.linspace(0.0, 1, num=20)  # Threshold levels
    noise_levels = np.linspace(0.0, 1.0, num=11)  # Noise intensity levels
    results = {threshold: {noise_level: None for noise_level in noise_levels} for threshold in thresholds}

    for noise_level in noise_levels:
        for threshold in thresholds:
            tp = fp = tn = fn = 0  # Initialize counters for each threshold-noise level combination
            for pairs_file in pairs_files:
                pairs = read_pairs(pairs_file)
                if not pairs:
                    continue

                for file1, file2, is_same in pairs:
                    image1 = preprocess_image(file1)
                    image2 = preprocess_image(file2)
                    image1 = add_noise_to_image(image1, noise_level)
                    image2 = add_noise_to_image(image2, noise_level)

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
            accuracy = (tp + tn) / total_comparisons if total_comparisons else 0
            results[threshold][noise_level] = accuracy

    # Plotting
    save_directory = "one-noise-fullnoise-bothside_plot"
    os.makedirs(save_directory, exist_ok=True)
    plt.figure(figsize=(10, 8))
    for noise_level in noise_levels:
        accuracies = [results[threshold][noise_level] for threshold in thresholds]
        plt.plot(thresholds, accuracies, label=f'Noise Level {noise_level:.2f}')

    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Effect of Noise Addition to Eye Regions on Face Recognition Accuracy Across Thresholds')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_directory, 'accuracy_vs_threshold_by_noise_level.png')
    plt.savefig(save_path)
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
