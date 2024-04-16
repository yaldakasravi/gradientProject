import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from comet_ml import Experiment

# Initialize Comet ML experiment
experiment = Experiment(api_key="UuHTEgYku8q9Ww3n13pSEgC8d", project_name="swap-bothPersons_effect", workspace="enhancing-gradient")

# Define paths
model_path = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/models/GN_W0.5_S2_ArcFace_epoch16.h5'
dataset_dir = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files_base = dataset_dir
pairs_files = [os.path.join(pairs_files_base, f'pairs_{i:02}.txt') for i in range(1, 11)]

def preprocess_image(image_path, num_squares, square_size):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [112, 112])
    img = tf.cast(img, tf.float32) / 255.0

    # Select a random different person's image for swapping
    all_person_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and d != os.path.basename(os.path.dirname(image_path))]
    if not all_person_dirs:
        print("No other directories available for swapping.")
        return preprocess_input(img)

    chosen_dir = np.random.choice(all_person_dirs)
    chosen_dir_path = os.path.join(dataset_dir, chosen_dir)
    all_images = [f for f in os.listdir(chosen_dir_path) if os.path.isfile(os.path.join(chosen_dir_path, f))]
    if not all_images:
        print("No images found in the chosen directory for swapping.")
        return preprocess_input(img)

    swap_image_path = os.path.join(chosen_dir_path, np.random.choice(all_images))
    swap_img = tf.io.read_file(swap_image_path)
    swap_img = tf.image.decode_jpeg(swap_img, channels=3)
    swap_img = tf.image.resize(swap_img, [112, 112])
    swap_img = tf.cast(swap_img, tf.float32) / 255.0

    # Define mask region and apply swap
    mask_start = (112 - square_size) // 2
    img[mask_start:mask_start+square_size, :, :] = swap_img[mask_start:mask_start+square_size, :, :]

    return preprocess_input(img)

def prepare_dataset(file_path, num_squares, square_size):
    def parse_line(line):
        parts = tf.strings.split(line)
        return parts[0], parts[1], tf.strings.to_number(parts[2], tf.int32)

    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(parse_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x, y, label: (
        (preprocess_image(os.path.join(dataset_dir, x), num_squares, square_size),
         preprocess_image(os.path.join(dataset_dir, y), num_squares, square_size)), label),
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

    save_directory = "swapping-featureimportance"
    os.makedirs(save_directory, exist_ok=True)

    plt.figure(figsize=(10, 8))
    for mask_thickness in mask_thickness_levels:
        avg_accuracies = []
        for threshold in thresholds:
            accuracies = []
            for pairs_file in pairs_files:
                dataset = prepare_dataset(pairs_file, int(mask_thickness * 10), 20)  # Apply swaps to both images
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

    plt.title("Accuracy vs. Threshold for Different Mask Thicknesses with Both Images Swapped")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(save_directory, "accuracy_vs_threshold.png"))
    plt.close()

    experiment.end()

if __name__ == "__main__":
    main()

