import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from comet_ml import Experiment

# Initialize your Comet ML experiment
experiment = Experiment(api_key="UuHTEgYku8q9Ww3n13pSEgC8d", project_name="swap-differentPerson_effect", workspace="enhancing-gradient")

# Define paths
model_path = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/models/GN_W0.5_S2_ArcFace_epoch16.h5'
dataset_dir = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files_base = dataset_dir
pairs_files = [os.path.join(pairs_files_base, f'pairs_{i:02}.txt') for i in range(1, 11)]

def preprocess_image(image_path, apply_mask, num_squares, square_size):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [112, 112])
    img = tf.cast(img, tf.float32) / 255.0

    if apply_mask:
        for _ in range(num_squares):
            x1 = tf.random.uniform((), 0, 112 - square_size, dtype=tf.int32)
            y1 = tf.random.uniform((), 0, 112 - square_size, dtype=tf.int32)
            mask = tf.tensor_scatter_nd_update(
                img,
                indices=[[y1 + i, x1 + j] for i in range(square_size) for j in range(square_size)],
                updates=tf.zeros((square_size, square_size, 3))
            )
        img = mask
    return preprocess_input(img)

def load_and_prepare_dataset(file_path, num_squares, square_size):
    def parse_line(line):
        parts = tf.strings.split(line)
        return parts[0], parts[1], tf.cast(parts[2], tf.int32)

    def load_images(path1, path2, label):
        img1 = preprocess_image(os.path.join(dataset_dir, path1), True, num_squares, square_size)
        img2 = preprocess_image(os.path.join(dataset_dir, path2), False, 0, 0)  # Reference image unmodified
        return (img1, img2), label

    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(parse_line)
    dataset = dataset.map(load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

def evaluate_model(model, dataset):
    similarities, labels = [], []
    for (img1, img2), label in dataset:
        embedding1 = model.predict(img1)
        embedding2 = model.predict(img2)
        similarity = 1 - tf.reduce_sum(embedding1 * embedding2, axis=1) / (tf.norm(embedding1, axis=1) * tf.norm(embedding2, axis=1))
        similarities.extend(similarity.numpy())
        labels.extend(label.numpy())
    return np.array(labels), np.array(similarities)

def main():
    model = load_model(model_path)
    thresholds = np.linspace(0.3, 1, num=14)
    num_squares_range = range(1, 11)
    square_size = 20

    plt.figure(figsize=(10, 8))
    for num_squares in num_squares_range:
        all_accuracies = []
        for pairs_file in pairs_files:
            dataset = load_and_prepare_dataset(pairs_file, num_squares, square_size)
            labels, similarities = evaluate_model(model, dataset)
            accuracies = []
            for threshold in thresholds:
                predictions = similarities >= threshold
                accuracy = np.mean(predictions == labels)
                accuracies.append(accuracy)
            all_accuracies.append(accuracies)
        mean_accuracies = np.mean(all_accuracies, axis=0)
        plt.plot(thresholds, mean_accuracies, label=f'Num Squares {num_squares}')
    
    # Plotting
    save_directory = "threshold-swap-differentPerson-oneSide_plot"
    os.makedirs(save_directory, exist_ok=True)

    plt.title("Accuracy vs. Threshold for Different Numbers of Squares for one side")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_vs_threshold_comparison.png")
    plt.show()

    experiment.end()

if __name__ == "__main__":
    main()

