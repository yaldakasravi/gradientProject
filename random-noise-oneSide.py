from comet_ml import Experiment

# Initialize your Comet ML experiment here
experiment = Experiment(api_key="UuHTEgYku8q9Ww3n13pSEgC8d", project_name="random-square-masking_effect", workspace="enhancing-gradient")

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import random

# Parameters
model_path = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/models/GN_W0.5_S2_ArcFace_epoch16.h5'
dataset_dir = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files = [os.path.join(dataset_dir, f'pairs_{i:02}.txt') for i in range(1, 11)]
noise_levels = np.linspace(0.0, 1.0, num=11)  # Intensity levels of noise
# Load the model
model = load_model(model_path)

"""
import random

def apply_random_mask(img_array, num_squares, square_size):
    
    Applies a given number of square masks randomly on the image.

    :param img_array: NumPy array of the image.
    :param num_squares: The number of squares to apply.
    :param square_size: The size of each square.
    
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
            # Apply mask to the first image only
            embedding1_same = get_embedding(model, img1_path_same, num_squares=num_squares, square_size=square_size)
            embedding2_same = get_embedding(model, img2_path_same, num_squares=0, square_size=square_size)  # No mask
            similarity_same = get_cosine_similarity(embedding1_same, embedding2_same)
            similarities.append(similarity_same)
            labels.append(1)  # Same person

            # Second pair (different people)
            img1_path_diff = os.path.join(dataset_dir, lines[i+2])
            img2_path_diff = os.path.join(dataset_dir, lines[i+3])
            # Apply mask to the first image only
            embedding1_diff = get_embedding(model, img1_path_diff, num_squares=num_squares, square_size=square_size)
            embedding2_diff = get_embedding(model, img2_path_diff, num_squares=0, square_size=square_size)  # No mask
            similarity_diff = get_cosine_similarity(embedding1_diff, embedding2_diff)
            similarities.append(similarity_diff)
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

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    with tf.device('/GPU:1'):
        model = load_model(model_path)
        metrics_summary = {}

        # Define ranges for number of squares and square size
        num_squares_range = range(1, 11)  # example: 1 to 10 squares
        square_size = 20  # example: each square is 20x20 pixels
        avg_metrics = {squares: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for squares in num_squares_range}

        for th in np.linspace(0.3,1,num=14):
            for num_squares in num_squares_range:
                metrics_results = []

                pair_acc = []
                pair_precision = []
                pair_recall = []
                pair_f1 = []

                for pairs_file in pairs_files:
                    pairs_file_path = os.path.join(pairs_files_base, pairs_file)
                    labels, similarities = evaluate_lfw(model, dataset_dir, pairs_file_path, num_squares=num_squares, square_size=square_size)
                    metrics = calculate_metrics(labels, similarities, th)
                    metrics_results.append(metrics)

                    # Extract metrics from the returned dictionary
                    accuracy = metrics['accuracy']
                    precision = metrics['precision']
                    recall = metrics['recall']
                    f1 = metrics['f1']

                    pair_acc.append(accuracy)
                    pair_precision.append(precision)
                    pair_recall.append(recall)
                    pair_f1.append(f1)
                   
                    # Print summary for the current pairs file
                    print(f"Summary for {pairs_file} with {num_squares} masks {th} threshold:")
                    print(f"  Accuracy: {accuracy:.4f}")
                    print(f"  Precision: {precision:.4f}")
                    print(f"  Recall: {recall:.4f}")
                    print(f"  F1 Score: {f1:.4f}\n")
            # Calculate average metrics for this noise level
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    metric_values = [m[metric] for m in metrics_results]
                    avg_metrics[num_squares][metric] = np.mean(metric_values)

                #avg_metrics = {metric: np.mean([res[metric] for res in metrics_results]) for metric in ['accuracy', 'precision', 'recall', 'f1']}
                #metrics_summary[num_squares] = avg_metrics
                # Log results to Comet ML
                experiment.log_metrics(avg_metrics, prefix=f"masks_{num_squares}")

            save_directory = "threshold-random-square-masking_plot"
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            # Plot results across different levels of masking
            for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
                plt.figure()
                x = list(metrics_summary.keys())
                y = [avg_metrics[num_squares][metric_name] for num_squares in num_squares_range]
                plt.plot(x, y, marker='o', linestyle='-')
                plt.title(f"{metric_name.capitalize()} vs. Number of Masks")
                plt.xlabel("Number of Masks")
                plt.ylabel(metric_name.capitalize())
                plt.grid(True)
                plt.savefig(os.path.join(save_directory, f"{metric_name}_across_pairs_files_at_{th:.2f}_threshold.png"))
                plt.close()

            experiment.end()

if __name__ == "__main__":
    main()

def main():
    with tf.device('/GPU:1'):
        model = load_model(model_path)
        thresholds = np.linspace(0.3, 1, num=14)
        num_squares_range = range(1, 11)  # Example: 1 to 10 squares
        square_size = 20  # Example: each square is 20x20 pixels

        # Initialize dictionary to store average accuracy for each number of squares across thresholds
        avg_accuracy_per_num_squares = {num_squares: [] for num_squares in num_squares_range}

        for th in thresholds:
            avg_metrics = {squares: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for squares in num_squares_range}

            for num_squares in num_squares_range:
                metrics_results = []

                for pairs_file in pairs_files:
                    pairs_file_path = os.path.join(pairs_files_base, pairs_file)
                    labels, similarities = evaluate_lfw(model, dataset_dir, pairs_file_path, num_squares=num_squares, square_size=square_size)
                    metrics = calculate_metrics(labels, similarities, th)
                    metrics_results.append(metrics)

                # Calculate average metrics for this configuration
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    metric_values = [m[metric] for m in metrics_results]
                    avg_metrics[num_squares][metric] = np.mean(metric_values)
                # Correctly print the summary using the avg_metrics for the current noise_factor and threshold
                print(f"  Accuracy: {avg_metrics[num_squares]['accuracy']:.4f}")
                print(f"  Precision: {avg_metrics[num_squares]['precision']:.4f}")
                print(f"  Recall: {avg_metrics[num_squares]['recall']:.4f}")
                print(f"  F1 Score: {avg_metrics[num_squares]['f1']:.4f}\n")
            
            # Store average accuracy for each number of squares for this threshold
            for num_squares in num_squares_range:
                avg_accuracy_per_num_squares[num_squares].append(avg_metrics[num_squares]['accuracy'])

        save_directory = "threshold-random-square-masking-oneSide_plot"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Plotting
        plt.figure()
        for num_squares in num_squares_range:
            plt.plot(thresholds, avg_accuracy_per_num_squares[num_squares], marker='o', linestyle='-', label=f'{num_squares} Squares')

        plt.title("Accuracy vs. Threshold for different Numbers of Squares one side")
        plt.xlabel("Threshold")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_directory, "accuracy_vs_threshold_for_number_of_squares-oneSide.png"))
        plt.close()

        # Note: Remember to end your experiment here if you're using a system like Comet ML for logging
        # experiment.end()

if __name__ == "__main__":
    main()
"""    
#using dataloader

def preprocess_image(image_path):
    image = Image.open(image_path).resize((112, 112))
    image = np.array(image, dtype='float32')
    image /= 255.0  # Normalize
    return image

def add_noise_to_image(image, noise_level):
    # Adding Gaussian noise
    mean = 0
    var = noise_level
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (112, 112, 3))  # Assuming RGB image
    noisy_image = np.clip(image + gaussian, 0, 1)  # Clipping to maintain valid pixel range
    return noisy_image

def calculate_similarity(image1, image2):
    emb1 = model.predict(np.expand_dims(image1, axis=0))
    emb2 = model.predict(np.expand_dims(image2, axis=0))
    similarity = np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity

def read_pairs(pairs_file):
    pairs = []
    with open(pairs_file, "r") as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                person1_image1 = lines[i].strip()
                person1_image2 = lines[i + 1].strip()
                if person1_image1 and person1_image2:
                    pairs.append((person1_image1, person1_image2, True))
    return pairs

def main():
    results = {level: [] for level in noise_levels}
    for level in noise_levels:
        accuracies = []
        for pairs_file in pairs_files:
            pairs = read_pairs(pairs_file)
            if not pairs:
                print(f"No valid pairs found in {pairs_file}.")
                continue
            tp = fp = tn = fn = 0
            for file1, file2, is_same in pairs:
                image1 = preprocess_image(os.path.join(dataset_dir, file1))
                image2 = preprocess_image(os.path.join(dataset_dir, file2))

                # Add noise to the first image only
                image1 = add_noise_to_image(image1, level)

                similarity = calculate_similarity(image1, image2)
                is_positive_match = similarity > 0.5  # Arbitrary threshold for simplicity
                if is_positive_match and is_same:
                    tp += 1
                elif is_positive_match and not is_same:
                    fp += 1
                elif not is_positive_match and not is_same:
                    tn += 1
                elif not is_positive_match and is_same:
                    fn += 1

            total_comparisons = tp + fp + tn + fn
            if total_comparisons == 0:
                accuracy = 0  # Append 0 to avoid division by zero
            else:
                accuracy = (tp + tn) / total_comparisons
            accuracies.append(accuracy)
        results[level] = np.mean(accuracies)

    # Plotting
    save_directory = "noise-intensity-oneSide_plot"
    os.makedirs(save_directory, exist_ok=True)
    plt.figure(figsize=(10, 8))
    for level, accuracy in results.items():
        plt.plot(noise_levels, [results[lv] for lv in noise_levels], 'o-', label=f'Noise Level {level:.2f}')
    plt.xlabel('Noise Level')
    plt.ylabel('Accuracy')
    plt.title('Effect of Noise Addition on Face Authentication Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_directory, 'accuracy_vs_noise_level.png'))
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
