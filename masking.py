import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from scipy.spatial import distance
from comet_ml import Experiment

# Initialize your Comet ML experiment here
experiment = Experiment(api_key="UuHTEgYku8q9Ww3n13pSEgC8d", project_name="masking_effect", workspace="enhancing-gradient")

# Define paths
model_path = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/models/GN_W0.5_S2_ArcFace_epoch16.h5'
dataset_dir = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files_base = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files = [f'pairs_{i:02}.txt' for i in range(1, 11)]  # Adjust the range as needed

#very slow 
"""
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
    with tf.device('/GPU:1'):
        model = load_model(model_path)
        thresholds = np.linspace(0.3, 1, num=14)
        mask_thickness_levels = np.linspace(0.1, 1.0, num=10)

        # Initialize dictionary to store average accuracy for each mask thickness level across thresholds
        avg_accuracy_per_thickness_level = {thickness: [] for thickness in mask_thickness_levels}

        for th in thresholds:
            avg_metrics = {thickness: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for thickness in mask_thickness_levels}

            for mask_thickness in mask_thickness_levels:
                all_metrics = []

                for pairs_file in pairs_files:
                    pairs_file_path = os.path.join(pairs_files_base, pairs_file)
                    y_true, y_pred_scores = evaluate_lfw(model, dataset_dir, pairs_file_path, mask_thickness)
                    metrics = calculate_metrics(y_true, y_pred_scores, th)
                    all_metrics.append(metrics)

                # Calculate average metrics for this mask thickness level
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    metric_values = [m[metric] for m in all_metrics]
                    avg_metrics[mask_thickness][metric] = np.mean(metric_values)
                # Correctly print the summary using the avg_metrics for the current noise_factor and threshold
                print(f"  Accuracy: {avg_metrics[mask_thickness]['accuracy']:.4f}")
                print(f"  Precision: {avg_metrics[mask_thickness]['precision']:.4f}")
                print(f"  Recall: {avg_metrics[mask_thickness]['recall']:.4f}")
                print(f"  F1 Score: {avg_metrics[mask_thickness]['f1']:.4f}\n")
            
            # Store average accuracy for each mask thickness level for this threshold
            for mask_thickness in mask_thickness_levels:
                avg_accuracy_per_thickness_level[mask_thickness].append(avg_metrics[mask_thickness]['accuracy'])

        # Plotting
        save_directory = "threshold-black-masking_plot"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        plt.figure()
        for mask_thickness in mask_thickness_levels:
            plt.plot(thresholds, avg_accuracy_per_thickness_level[mask_thickness], marker='o', linestyle='-', label=f'Thickness {mask_thickness:.1f}')

        plt.title("Accuracy vs. Threshold for different Mask Thicknesses")
        plt.xlabel("Threshold")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_directory, "accuracy_vs_threshold_for_mask_thicknesses.png"))
        plt.close()

if __name__ == "__main__":
    main()
"""
#using dataloader 
def preprocess_image(image_path, mask_thickness):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [112, 112])
    img = tf.cast(img, tf.float32) / 255.0

    # Calculate the number of rows to mask based on the mask thickness
    num_rows_to_mask = tf.cast(112 * mask_thickness, tf.int32)
    start_row = (112 - num_rows_to_mask) // 2
    end_row = start_row + num_rows_to_mask

    # Create a mask and apply it
    mask = tf.concat([
        tf.ones((start_row, 112, 3)),
        tf.zeros((num_rows_to_mask, 112, 3)),
        tf.ones((112 - end_row, 112, 3))
    ], axis=0)
    img *= mask
    return preprocess_input(img)

def compute_cosine_similarity(embeddings1, embeddings2):
    dot_product = tf.reduce_sum(embeddings1 * embeddings2, axis=1)
    norm_product = tf.norm(embeddings1, axis=1) * tf.norm(embeddings2, axis=1)
    return 1 - dot_product / norm_product

def prepare_dataset(pairs_file_path, mask_thickness):
    def parse_line(line):
        parts = tf.strings.split(line)
        img1_path = tf.strings.join([dataset_dir, parts[0]])
        img2_path = tf.strings.join([dataset_dir, parts[1]])
        label = tf.strings.to_number(parts[2], tf.int32)
        return img1_path, img2_path, label

    dataset = tf.data.TextLineDataset(pairs_file_path)
    dataset = dataset.map(parse_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x, y, label: (
        (preprocess_image(x, mask_thickness), preprocess_image(y, 0)), label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

def evaluate_model(model, dataset):
    embeddings1 = []
    embeddings2 = []
    labels = []

    for (img1, img2), label in dataset:
        embeddings1.append(model.predict(img1))
        embeddings2.append(model.predict(img2))
        labels.append(label)

    embeddings1 = tf.concat(embeddings1, axis=0)
    embeddings2 = tf.concat(embeddings2, axis=0)
    labels = tf.concat(labels, axis=0)

    similarities = compute_cosine_similarity(embeddings1, embeddings2)
    return labels, similarities

def main():
    model = load_model(model_path)
    thresholds = np.linspace(0.3, 1, num=14)
    mask_thickness_levels = np.linspace(0.1, 1.0, num=10)

    save_directory = "masking_plot"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    plt.figure()
    for mask_thickness in mask_thickness_levels:
        accuracies = []
        for threshold in thresholds:
            all_accuracies = []
            for pairs_file in pairs_files:
                dataset = prepare_dataset(pairs_file, mask_thickness)
                labels, similarities = evaluate_model(model, dataset)
                predictions = similarities >= threshold
                accuracy = tf.reduce_mean(tf.cast(predictions == labels, tf.float32)).numpy()
                all_accuracies.append(accuracy)
            accuracies.append(np.mean(all_accuracies))
        plt.plot(thresholds, accuracies, marker='o', linestyle='-', label=f'Thickness {mask_thickness:.1f}')

    plt.title("Accuracy vs. Threshold for Different Mask Thicknesses")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_directory, "accuracy_vs_threshold.png"))
    plt.close()

    experiment.end()

if __name__ == "__main__":
    main()
