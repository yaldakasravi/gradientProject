import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from scipy.spatial.distance import cosine
from comet_ml import Experiment

# Initialize Comet ML experiment
experiment = Experiment(api_key="UuHTEgYku8q9Ww3n13pSEgC8d", project_name="swap-differentPerson_effect", workspace="enhancing-gradient")

# Define paths
model_path = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/models/GN_W0.5_S2_ArcFace_epoch16.h5'
dataset_dir = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files_base = dataset_dir
pairs_files = [os.path.join(pairs_files_base, f'pairs_{i:02}.txt') for i in range(1, 11)]

def preprocess_image(image_path, apply_mask, num_squares=0, square_size=20):
    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [112, 112])
        img = tf.cast(img, tf.float32) / 255.0

        if apply_mask and num_squares > 0:
            # Create mask dynamically based on the number of squares and their size
            for _ in range(num_squares):
                x1 = tf.random.uniform((), 0, img.shape[1] - square_size, dtype=tf.int32)
                y1 = tf.random.uniform((), 0, img.shape[0] - square_size, dtype=tf.int32)
                mask = tf.pad(tensor=tf.ones((square_size, square_size, 3), dtype=tf.float32),
                              paddings=[[y1, img.shape[0] - y1 - square_size], 
                                        [x1, img.shape[1] - x1 - square_size], [0, 0]],
                              mode="CONSTANT", constant_values=0)
                img *= mask
        return preprocess_input(img)
    except Exception as e:
        print(f"Failed to process image {image_path}: {e}")
        return None

def prepare_dataset(file_path, num_squares, square_size):
    def parse_line(line):
        parts = tf.strings.split(line)
        return parts[0], parts[1], tf.strings.to_number(parts[2], tf.int32)

    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(parse_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x, y, label: (
        (preprocess_image(os.path.join(dataset_dir, x), True, num_squares, square_size),
         preprocess_image(os.path.join(dataset_dir, y), False)), label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

def compute_similarity(embeddings1, embeddings2):
    dot_product = tf.reduce_sum(embeddings1 * embeddings2, axis=1)
    norm_product = tf.norm(embeddings1, axis=1) * tf.norm(embeddings2, axis=1)
    return 1 - dot_product / norm_product

def main():
    model = load_model(model_path)
    thresholds = np.linspace(0.3, 1, num=14)
    mask_thickness_levels = np.linspace(0.1, 1.0, num=10)

    save_directory = "threshold-swap-samePerson-oneSide_plot"
    os.makedirs(save_directory, exist_ok=True)

    plt.figure(figsize=(10, 8))
    for mask_thickness in mask_thickness_levels:
        avg_accuracies = []
        for threshold in thresholds:
            accuracies = []
            for pairs_file in pairs_files:
                dataset = prepare_dataset(pairs_file, int(mask_thickness * 10), 20)  # Example: mask_thickness * 10 squares
                embeddings1, embeddings2, labels = [], [], []
                for (img1, img2), label in dataset:
                    emb1 = model(img1, training=False)
                    emb2 = model(img2, training=False)
                    embeddings1.append(emb1)
                    embeddings2.append(emb2)
                    labels.append(label)
                embeddings1 = tf.concat(embeddings1, axis=0)
                embeddings2 = tf.concat(embeddings2, axis=0)
                labels = tf.concat(labels, axis=0)
                similarities = compute_similarity(embeddings1, embeddings2)
                predictions = similarities >= threshold
                accuracy = tf.reduce_mean(tf.cast(predictions == labels, tf.float32))
                accuracies.append(accuracy.numpy())
            avg_accuracies.append(np.mean(accuracies))
        plt.plot(thresholds, avg_accuracies, marker='o', linestyle='-', label=f'Mask Thickness {mask_thickness:.2f}')
        
    # Setup for plotting

    plt.title("Accuracy vs. Threshold for Different Mask Thicknesses one side")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(save_directory, "accuracy_vs_threshold.png"))
    plt.close()

    experiment.end()

if __name__ == "__main__":
    main()

