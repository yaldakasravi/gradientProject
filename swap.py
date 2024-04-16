import numpy as np
import os
import random
from comet_ml import Experiment
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import tensorflow as tf

# Initialize your Comet ML experiment here
experiment = Experiment(api_key="UuHTEgYku8q9Ww3n13pSEgC8d", project_name="swap-differentPerson_effect", workspace="enhancing-gradient")
#experiment = Experiment(api_key="YourCometMLAPIKey", project_name="swap-samePerson_effect", workspace="enhancing-gradient")

# Define paths
model_path = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/models/GN_W0.5_S2_ArcFace_epoch16.h5'
dataset_dir = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files_base = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files = [f'pairs_{i:02}.txt' for i in range(1, 11)]  # Adjust the range as needed

#ver slow 
#different person 
"""
def load_and_preprocess_image(image_path, mask_thickness, dataset_dir):
    # Load the target image
    img = image.load_img(image_path, target_size=(112, 112))
    img_array = image.img_to_array(img)

    # Extract person's name from the image path to avoid choosing the same person
    current_person_name = os.path.basename(os.path.dirname(image_path))

    # Get all person directories in the dataset
    all_person_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    # Remove the current person's directory from the list
    all_person_dirs.remove(current_person_name)

    # Select a random different person's directory
    different_person_dir = random.choice(all_person_dirs)

    # Construct the path to the random different person's directory
    different_person_path = os.path.join(dataset_dir, different_person_dir)

    # Get all images of the randomly selected different person
    different_person_images = os.listdir(different_person_path)

    # Randomly select an image from the different person
    different_person_image = random.choice(different_person_images)
    different_person_image_path = os.path.join(different_person_path, different_person_image)

    # Load the randomly selected different person's image
    different_img = image.load_img(different_person_image_path, target_size=(112, 112))
    different_img_array = image.img_to_array(different_img)

    # Define the number of rows to swap based on the mask thickness
    num_rows_to_swap = int(112 * mask_thickness)

    # Swap a horizontal strip across the center of the image
    start_row = (112 - num_rows_to_swap) // 2
    end_row = start_row + num_rows_to_swap
    img_array[start_row:end_row, :, :] = different_img_array[start_row:end_row, :, :]

    # Preprocess the image for the model
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)

def get_embedding(model, image_path, mask_thickness):
    processed_image = load_and_preprocess_image(image_path, mask_thickness, dataset_dir)
    return model.predict(processed_image)

def get_cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1.flatten(), embedding2.flatten())


def evaluate_lfw(model, dataset_dir, pairs_file_path, mask_thickness):
    similarities, labels = [], []
    with open(pairs_file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        # Assuming each block of 4 lines in the pairs file corresponds to two pairs of images
        for i in range(0, len(lines), 4):
            # First pair (same person)
            img1_path_same = os.path.join(dataset_dir, lines[i])
            img2_path_same = os.path.join(dataset_dir, lines[i+1])
            #print("Loading same pair:", img1_path_same, img2_path_same)
            embedding1_same = get_embedding(model, img1_path_same, mask_thickness)
            embedding2_same = get_embedding(model, img2_path_same, mask_thickness)
            similarity_same = get_cosine_similarity(embedding1_same, embedding2_same)
            similarities.append(similarity_same)
            labels.append(1)  # Same person

            # Second pair (different people)
            img1_path_diff = os.path.join(dataset_dir, lines[i+2])
            img2_path_diff = os.path.join(dataset_dir, lines[i+3])
            embedding1_diff = get_embedding(model, img1_path_diff, mask_thickness)
            embedding2_diff = get_embedding(model, img2_path_diff, mask_thickness)
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

                # Calculate and store the average metrics for this mask thickness level
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    metric_values = [m[metric] for m in all_metrics]
                    avg_metrics[mask_thickness][metric] = np.mean(metric_values)
                # Correctly print the summary using the avg_metrics for the current noise_factor and threshold
                print(f"  Accuracy: {avg_metrics[mask_thickness]['accuracy']:.4f}")
                print(f"  Precision: {avg_metrics[mask_thickness]['precision']:.4f}")
                print(f"  Recall: {avg_metrics[mask_thickness]['recall']:.4f}")
                print(f"  F1 Score: {avg_metrics[mask_thickness]['f1']:.4f}\n")
            
            # Accumulate average accuracy for plotting
            for mask_thickness in mask_thickness_levels:
                avg_accuracy_per_thickness_level[mask_thickness].append(avg_metrics[mask_thickness]['accuracy'])

        # Prepare for plotting
        save_directory = "threshold-swap-differentPerson_plot"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        plt.figure()
        for mask_thickness in mask_thickness_levels:
            plt.plot(thresholds, avg_accuracy_per_thickness_level[mask_thickness], marker='o', linestyle='-', label=f'Thickness {mask_thickness:.1f}')

        plt.title("Accuracy vs. Threshold for Different Mask Thicknesses")
        plt.xlabel("Threshold")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_directory, "accuracy_vs_threshold_for_mask_thicknesses.png"))
        plt.close()

if __name__ == "__main__":
    main()
"""

#using dataloader to make it faster

# TensorFlow compatible image loading and preprocessing

def find_random_directory_with_images(base_dir, exclude_dir):
    all_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d != exclude_dir]
    random.shuffle(all_dirs)
    for chosen_dir in all_dirs:
        chosen_dir_path = os.path.join(base_dir, chosen_dir)
        all_images = [f for f in os.listdir(chosen_dir_path) if f.endswith('.jpg')]
        if all_images:
            random_image = random.choice(all_images)
            return os.path.join(chosen_dir_path, random_image)
    return None

def load_and_preprocess_image(image_path, dataset_dir):
    current_dir = os.path.basename(os.path.dirname(image_path))
    random_image_path = find_random_directory_with_images(dataset_dir, current_dir)
    if random_image_path is None:
        print("Failed to find a suitable random image.")
        return None
    
    try:
        # Load and process the original image
        img = image.load_img(image_path, target_size=(112, 112))
        img_array = image.img_to_array(img)

        # Load and process the random image
        random_img = image.load_img(random_image_path, target_size=(112, 112))
        random_img_array = image.img_to_array(random_img)

        # Combine images
        img_array[56:,:] = random_img_array[56:,:]  # Swap half the image
        return preprocess_input(np.expand_dims(img_array, axis=0))
    except Exception as e:
        print(f"Error processing images: {e}")
        return None

def evaluate_lfw(model, dataset_dir, pairs_files):
    labels, similarities = [], []
    for pairs_file in pairs_files:
        pairs_path = os.path.join(pairs_files_base, pairs_file)
        with open(pairs_path, 'r') as file:
            lines = file.read().splitlines()
        
        for i in range(0, len(lines), 2):
            img1_path = os.path.join(dataset_dir, lines[i])
            img2_path = os.path.join(dataset_dir, lines[i+1])
            img1 = load_and_preprocess_image(img1_path, dataset_dir)
            img2 = load_and_preprocess_image(img2_path, dataset_dir)

            if img1 is None or img2 is None:
                print("Skipping pair due to processing error.")
                continue
            
            embedding1 = model.predict(img1)
            embedding2 = model.predict(img2)
            similarity = 1 - cosine(embedding1.flatten(), embedding2.flatten())
            similarities.append(similarity)
            labels.append(1 if "same" in pairs_file else 0)  # Assuming file naming convention to determine labels
            
    return np.array(labels), np.array(similarities)

def calculate_metrics(labels, similarities):
    threshold = 0.5  # Example threshold
    predictions = similarities >= threshold
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def main():
    model = load_model(model_path)
    thresholds = np.linspace(0.3, 1, num=14)
    mask_thickness_levels = np.linspace(0.1, 1.0, num=10)

    all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    metrics_per_thickness = {thickness: {metric: [] for metric in all_metrics} for thickness in mask_thickness_levels}

    # Evaluate each mask thickness
    for mask_thickness in mask_thickness_levels:
        labels, similarities = evaluate_lfw(model, dataset_dir, pairs_files, mask_thickness)
        # Collect metrics for each threshold
        for threshold in thresholds:
            metrics = calculate_metrics(labels, similarities, threshold)
            for metric in all_metrics:
                metrics_per_thickness[mask_thickness][metric].append(metrics[metric])

    # Plotting
    save_directory = "threshold-swap-differentPerson_plot"
    os.makedirs(save_directory, exist_ok=True)

    plt.figure(figsize=(14, 10))
    for metric in all_metrics:
        plt.clf()  # Clear the current figure
        for mask_thickness, metrics in metrics_per_thickness.items():
            plt.plot(thresholds, metrics[metric], marker='o', linestyle='-', label=f'Mask Thickness {mask_thickness:.2f}')
        
        plt.title(f'{metric.capitalize()} vs. Threshold for Various Mask Thicknesses')
        plt.xlabel('Threshold')
        plt.ylabel(metric.capitalize())
        plt.legend(title='Mask Thickness', loc='best')
        plt.grid(True)
        save_path = os.path.join(save_directory, f'{metric}_vs_threshold.png')
        plt.savefig(save_path)
        print(f'Plot saved to {save_path}')
if __name__ == "__main__":
    main()
