import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import random
# Initialize your Comet ML experiment here
#experiment = Experiment(api_key="UuHTEgYku8q9Ww3n13pSEgC8d", project_name="masking_effect", workspace="enhancing-gradient")

# Parameters
model_path = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/models/GN_W0.5_S2_ArcFace_epoch16.h5'
dataset_dir = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files = [os.path.join(dataset_dir, f'pairs_{i:02}.txt') for i in range(1, 11)]
thresholds = np.linspace(0.3, 1, num=14)
eye_mask_levels = np.linspace(0, 1, num=10)  # Intensity levels of eye mask

# Load the model
model = load_model(model_path)
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

#using dataloader to be faster 

def preprocess_image(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Expected file but got directory or non-existent path: {image_path}")
    image = Image.open(image_path).resize((112, 112))
    image = np.array(image, dtype='float32')
    image /= 255.0  # Normalize
    return image

def mask_eyes(image, level):
    eye_width = 20  # Width of the eye region
    eye_height = 10  # Height of the eye region
    left_eye_center = (34, 56)  # (x, y) positions
    right_eye_center = (78, 56)
    mask_width = int(eye_width * level / 2)
    image[left_eye_center[1]-eye_height//2:left_eye_center[1]+eye_height//2,
          left_eye_center[0]-mask_width:left_eye_center[0]+mask_width] = 0
    image[right_eye_center[1]-eye_height//2:right_eye_center[1]+eye_height//2,
          right_eye_center[0]-mask_width:right_eye_center[0]+mask_width] = 0
    return image

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
                file1 = os.path.join(dataset_dir, lines[i].strip())
                file2 = os.path.join(dataset_dir, lines[i + 1].strip())
                if os.path.isfile(file1) and os.path.isfile(file2):
                    pairs.append((file1, file2, True))
    return pairs

def main():
    results = {level: {threshold: [] for threshold in thresholds} for level in eye_mask_levels}
    for level in eye_mask_levels:
        for threshold in thresholds:
            accuracies = []
            for pairs_file in pairs_files:
                pairs = read_pairs(pairs_file)
                if not pairs:
                    continue
                tp = fp = tn = fn = 0
                for file1, file2, is_same in pairs:
                    image1 = preprocess_image(file1)
                    image2 = preprocess_image(file2)
                    # Apply masking to both images
                    image1 = mask_eyes(image1, level)
                    1
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
                if total_comparisons == 0:
                    accuracy = 0
                else:
                    accuracy = (tp + tn) / total_comparisons
                accuracies.append(accuracy)
            results[level][threshold] = np.mean(accuracies)

    # Plotting
    save_directory = "threshold-mask-oneside_plot"
    os.makedirs(save_directory, exist_ok=True)

    plt.figure(figsize=(10, 8))
    for level, accuracies_by_threshold in results.items():
        thresholds_list = list(accuracies_by_threshold.keys())
        accuracies_list = [accuracies_by_threshold[th] for th in thresholds_list]
        plt.plot(thresholds_list, accuracies_list, label=f'Mask Level {level:.2f}')

    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Effect of Eye Masking at Different Levels on Both Images')
    plt.legend(title='Mask Level')
    plt.grid(True)
    save_path = os.path.join(save_directory, 'accuracy_vs_thresholds_by_mask_level.png')
    plt.savefig(save_path)
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()

