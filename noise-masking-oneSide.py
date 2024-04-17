from comet_ml import Experiment
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
from utils import read_pairs

# Initialize your Comet ML experiment here
experiment = Experiment(api_key="UuHTEgYku8q9Ww3n13pSEgC8d", project_name="masking_effect-noise-oneside", workspace="enhancing-gradient")

# Define paths
model_path = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/models/GN_W0.5_S2_ArcFace_epoch16.h5'
dataset_dir = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files_base = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
#pairs_files = [os.path.join(dataset_dir, f'pairs_{i:02}.txt') for i in range(1, 11)]
pairs_files = [os.path.join(dataset_dir, f'pairs_{i:02}.txt') for i in range(1, 2)]
# Load the model
model = load_model(model_path)
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
def preprocess_image(image_path):
    image = Image.open(image_path).resize((112, 112))
    image = np.array(image, dtype='float32')
    image /= 255.0  # Normalize
    return image

def add_noise_to_eyes(image, level):
    eye_width = 20  # Width of the eye region
    eye_height = 10  # Height of the eye region
    left_eye_center = (34, 56)  # (x, y) positions
    right_eye_center = (78, 56)

    # Generate noise
    noise_intensity = level * 255  # Scale by 255 as pixel values range from 0 to 255
    # Ensure noise is generated for each color channel
    noise = np.random.normal(loc=0.0, scale=noise_intensity, size=(eye_height, eye_width, 3))

    # Apply noise to the eye regions
    for center in [left_eye_center, right_eye_center]:
        x_start = center[0] - eye_width // 2
        y_start = center[1] - eye_height // 2
        image[y_start:y_start + eye_height, x_start:x_start + eye_width] += noise
        # Ensure pixel values remain within [0, 255]
        np.clip(image, 0, 255, out=image)

    return image

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
                    image1 = add_noise_to_eyes(image1, noise_level)
                    #image2 = add_noise_to_eyes(image2, noise_level)
                    
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
    save_directory = "one-noise-masking-oneside_plot"
    os.makedirs(save_directory, exist_ok=True)
    plt.figure(figsize=(10, 8))
    for noise_level in noise_levels:
        accuracies = [results[threshold][noise_level] for threshold in thresholds]
        plt.plot(thresholds, accuracies, label=f'Noise Level {noise_level:.2f}')
    
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Effect of Noise Addition to Eye Regions in one iamge on Face Recognition Accuracy Across Thresholds')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_directory, 'accuracy_vs_threshold_by_noise_level.png')
    plt.savefig(save_path)
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
