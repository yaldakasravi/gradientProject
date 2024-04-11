import tensorflow as tf
import numpy as np
import os
import random
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

# Initialize your Comet ML experiment here
experiment = Experiment(api_key="UuHTEgYku8q9Ww3n13pSEgC8d", project_name="random-square-masking_effect", workspace="enhancing-gradient")

# Define paths
model_path = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/models/GN_W0.5_S2_ArcFace_epoch16.h5'
dataset_dir = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files_base = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files = [f'pairs_{i:02}.txt' for i in range(1, 11)]  # Adjust the range as needed

def apply_random_mask(img_array, num_squares, square_size):
    h, w, _ = img_array.shape
    for _ in range(num_squares):
        x1 = random.randint(0, w - square_size)
        y1 = random.randint(0, h - square_size)
        img_array[y1:y1+square_size, x1:x1+square_size, :] = 0
    return img_array

def tf_preprocess_image(image_path, num_squares=0, square_size=20):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [112, 112])
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)

    if num_squares > 0:
        img = tf.numpy_function(lambda x: apply_random_mask(x, num_squares, square_size), [img], tf.float32)
    return img

def parse_function(img_path1, img_path2, label):
    return (tf_preprocess_image(img_path1, num_squares=3, square_size=20), tf_preprocess_image(img_path2)), label

def load_pairs(dataset_dir, pairs_file_path):
    lines = tf.io.gfile.read_file(pairs_file_path).splitlines()
    img_paths1 = []
    img_paths2 = []
    labels = []

    for i in range(0, len(lines), 4):
        img_paths1.append(os.path.join(dataset_dir, lines[i]))
        img_paths2.append(os.path.join(dataset_dir, lines[i+1]))
        labels.append(1)  # Same person

        img_paths1.append(os.path.join(dataset_dir, lines[i+2]))
        img_paths2.append(os.path.join(dataset_dir, lines[i+3]))
        labels.append(0)  # Different people

    return tf.data.Dataset.from_tensor_slices((img_paths1, img_paths2, labels))

def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(32)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

def calculate_cosine_similarity(embeddings):
    # Assuming embeddings is a batch of pairs of embeddings
    embeddings1, embeddings2 = embeddings[:, 0, :], embeddings[:, 1, :]
    dot_product = tf.reduce_sum(embeddings1 * embeddings2, axis=1)
    norm_product = tf.norm(embeddings1, axis=1) * tf.norm(embeddings2, axis=1)
    cosine_similarity = dot_product / norm_product
    return 1 - cosine_similarity

def evaluate_lfw(model, dataset):
    similarities = []
    labels = []
    
    for (img1, img2), label in dataset:
        embeddings = model.predict(tf.stack([img1, img2], axis=0))  # Process pairs together
        similarity = calculate_cosine_similarity(embeddings)
        similarities.extend(similarity.numpy())
        labels.extend(label.numpy())
    
    return np.array(labels), np.array(similarities)


def main():
    thresholds = np.linspace(0.3, 1, num=14)
    num_squares_range = range(0, 11)  # Example: 0 to 10 squares, including no masking as a baseline
    square_size = 20  # Size of each square mask

    # Prepare the dataset
    dataset = load_pairs(dataset_dir, pairs_file_path)
    dataset = dataset.map(lambda img_path1, img_path2, label: parse_function(img_path1, img_path2, label, num_squares=0, square_size=square_size), 
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = configure_for_performance(dataset)

    # Dictionary to store average accuracy for each number of squares across thresholds
    accuracy_results = {num_squares: [] for num_squares in num_squares_range}

    for num_squares in num_squares_range:
        for threshold in thresholds:
            # Reconfigure dataset for current number of squares
            dataset = dataset.map(lambda img_path1, img_path2, label: parse_function(img_path1, img_path2, label, num_squares=num_squares, square_size=square_size), 
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
            
            labels, similarities = evaluate_lfw(model, dataset)
            predictions = similarities >= threshold
            accuracy = np.mean(predictions == labels)
            
            accuracy_results[num_squares].append(accuracy)

            print(f"Threshold: {threshold:.2f}, Num Squares: {num_squares}, Accuracy: {accuracy:.4f}")
        
    save_directory = "threshold-random-square-masking_plot"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Plotting
    plt.figure(figsize=(10, 8))
    for num_squares, accuracies in accuracy_results.items():
        plt.plot(thresholds, accuracies, marker='o', linestyle='-', label=f'{num_squares} Squares')

    plt.title("Accuracy vs. Threshold for Different Numbers of Masking Squares")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_vs_threshold_for_number_of_squares.png")
    plt.show()

if __name__ == "__main__":
    main()
