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
experiment = Experiment(api_key="YourCometMLAPIKey", project_name="swap-differentPerson_effect", workspace="enhancing-gradient")
#experiment = Experiment(api_key="YourCometMLAPIKey", project_name="swap-samePerson_effect", workspace="enhancing-gradient")

# Define paths
model_path = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/models/GN_W0.5_S2_ArcFace_epoch16.h5'
dataset_dir = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files_base = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files = [f'pairs_{i:02}.txt' for i in range(1, 11)]  # Adjust the range as needed

def load_and_preprocess_image(image_path, mask_thickness, dataset_dir):
    img = image.load_img(image_path, target_size=(112, 112))  # Load the image with the target size
    img_array = image.img_to_array(img)  # Convert to a NumPy array

    # Randomly select another image to swap with
    all_images = os.listdir(dataset_dir)
    random_image_path = os.path.join(dataset_dir, random.choice(all_images))
    random_img = image.load_img(random_image_path, target_size=(112, 112))
    random_img_array = image.img_to_array(random_img)

    # Define the number of rows to swap based on the mask thickness
    num_rows_to_swap = int(112 * mask_thickness)  # convert thickness proportion to number of rows

    # Swap a horizontal strip across the center of the image
    start_row = (112 - num_rows_to_swap) // 2
    end_row = start_row + num_rows_to_swap
    img_array[start_row:end_row, :, :] = random_img_array[start_row:end_row, :, :]  # Swap the pixel values

    # Preprocess the image for the model
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)

#for same person swap 
"""
def load_and_preprocess_image(image_path, mask_thickness, dataset_dir):
    # Load the target image
    img = image.load_img(image_path, target_size=(112, 112))
    img_array = image.img_to_array(img)

    # Extract directory and base filename without extension
    person_dir = os.path.dirname(image_path)
    base_filename = os.path.basename(image_path).split('.')[0]

    # Find other images of the same person
    other_images = [f for f in os.listdir(person_dir) if os.path.isfile(os.path.join(person_dir, f)) and f != base_filename + '.jpg']
    if not other_images:
        # If no other images are found, use the original image (consider raising an error or handling this case differently)
        swap_img_array = img_array
    else:
        # Randomly select another image of the same person for swapping
        swap_image_path = os.path.join(person_dir, random.choice(other_images))
        swap_img = image.load_img(swap_image_path, target_size=(112, 112))
        swap_img_array = image.img_to_array(swap_img)

    # Define the number of rows to swap based on the mask thickness
    num_rows_to_swap = int(112 * mask_thickness)

    # Swap a horizontal strip across the center of the image
    start_row = (112 - num_rows_to_swap) // 2
    end_row = start_row + num_rows_to_swap
    img_array[start_row:end_row, :, :] = swap_img_array[start_row:end_row, :, :]

    # Preprocess the image for the model
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)
"""
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
def main():
    model = load_model(model_path)
    metrics_results = []
    fixed_threshold = 0.65
    # Instead of noise_levels, define mask thickness levels (e.g., 10% to 100% of the image height)
    mask_thickness_levels = np.linspace(0.1, 1.0, num=10)

    avg_metrics = {thickness: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for thickness in mask_thickness_levels}

    for mask_thickness in mask_thickness_levels:
        all_metrics = []
        for pairs_file in pairs_files:
            pairs_file_path = os.path.join(pairs_files_base, pairs_file)
            y_true, y_pred_scores = evaluate_lfw(model, dataset_dir, pairs_file_path, mask_thickness)
            metrics = calculate_metrics(y_true, y_pred_scores, fixed_threshold)
            all_metrics.append(metrics)
             result = calculate_metrics(y_true, y_pred_scores, fixed_threshold)
            if isinstance(result, dict):
                all_metrics.append(result)
                accuracy = result['accuracy']
                precision = result['precision']
                recall = result['recall']
                f1 = result['f1']
            else:
                print("Unexpected result type:", type(result))
            # Extract metrics from the returned dictionary
            #accuracy = metrics['accuracy']
            #precision = metrics['precision']
            #recall = metrics['recall']
            #f1 = metrics['f1']

            # Print summary for the current pairs file
            print(f"Summary for {pairs_file} with {mask_thickness} :")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}\n")

        # Calculate average metrics for this noise level
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            metric_values = [m[metric] for m in all_metrics]
            avg_metrics[mask_thickness][metric] = np.mean(metric_values)
    save_directory = "swap-defferentPerson_plot"
    #save_directory = "swap-samePerson_plot"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
        plt.figure()
        thickness_factors = list(mask_thickness_levels)
        metric_values = [avg_metrics[thickness][metric_name] for thickness in thickness_factors]
        plt.plot(thickness_factors, metric_values, marker='o', linestyle='-')
        plt.title(f"{metric_name.capitalize()} vs. Mask Thickness")
        plt.xlabel("Mask Thickness")
        plt.ylabel(metric_name.capitalize())
        plt.grid(True)
        plt.savefig(os.path.join(save_directory, f"{metric_name}_vs_mask_thickness.png"))
        plt.close()
if __name__ == "__main__":
    main()

