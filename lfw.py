import numpy as np
import os
from comet_ml import Experiment
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import tensorflow as tf

# Assuming the Comet ML experiment is correctly initialized
experiment = Experiment(api_key="UuHTEgYku8q9Ww3n13pSEgC8d", project_name="threshold-analysis", workspace="enhancing-gradient")

# Define paths
model_path = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/models/GN_W0.5_S2_ArcFace_epoch16.h5'
dataset_dir = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files_base = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files = [f'pairs_{i:02}.txt' for i in range(1, 11)]  # Adjust the range as needed

def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(112, 112))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)

def get_embedding(model, image_path):
    processed_image = load_and_preprocess_image(image_path)
    return model.predict(processed_image)

def get_cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1.flatten(), embedding2.flatten())

def evaluate_lfw(model, dataset_dir, pairs_file_path):
    similarities, labels = [], []
    with open(pairs_file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        # Assuming each block of 4 lines in the pairs file corresponds to two pairs of images
        for i in range(0, len(lines), 4):
            # First pair (same person)
            img1_path_same = os.path.join(dataset_dir, lines[i])
            img2_path_same = os.path.join(dataset_dir, lines[i+1])
            #print("Loading same pair:", img1_path_same, img2_path_same)
            embedding1_same = get_embedding(model, img1_path_same)
            embedding2_same = get_embedding(model, img2_path_same)
            similarity_same = get_cosine_similarity(embedding1_same, embedding2_same)
            similarities.append(similarity_same)
            labels.append(1)  # Same person

            # Second pair (different people)
            img1_path_diff = os.path.join(dataset_dir, lines[i+2])
            img2_path_diff = os.path.join(dataset_dir, lines[i+3])
            embedding1_diff = get_embedding(model, img1_path_diff)
            embedding2_diff = get_embedding(model, img2_path_diff)
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
    
    return accuracy, precision, recall, f1

# Define the directory where you want to save the plots
plots_directory = " threshold-plots"

# Check if the directory exists, if not, create it
if not os.path.exists(plots_directory):
    os.makedirs(plots_directory)

def plot_and_save_metrics(metrics, pairs_file, thresholds):
    plt.figure(figsize=(10, 8))

    # Plot each metric in a subplot
    for i, (metric_name, values) in enumerate(metrics.items(), 1):
        plt.subplot(2, 2, i)
        plt.plot(thresholds, values, label=metric_name, marker='o')
        plt.title(f'{metric_name.capitalize()} over thresholds')
        plt.xlabel('Threshold')
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    # Use the plots_directory path when saving the figure
    plot_path = os.path.join(plots_directory, f"{pairs_file}_metrics.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()  # Close the figure to avoid display if running in a script

def main():
    model = load_model(model_path)
    # Define your thresholds here (convert percentages to decimal values)
    thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    for pairs_file in pairs_files:
        print(f"Evaluating {pairs_file}...")
        pairs_file_path = os.path.join(pairs_files_base, pairs_file)
        y_true, y_pred_scores = evaluate_lfw(model, dataset_dir, pairs_file_path)

        for threshold in thresholds:
            accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred_scores, threshold)
            # Print summary for each threshold
            print(f"Threshold: {threshold*100:.0f}%")
            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n")

        # You can still plot and save the metrics if you want, or remove this part if unnecessary
        metrics_results = {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1']}
        for threshold in thresholds:
            metrics = calculate_metrics(y_true, y_pred_scores, threshold)
            for metric_name, metric_value in zip(['accuracy', 'precision', 'recall', 'f1'], metrics):
                metrics_results[metric_name].append(metric_value)
        plot_and_save_metrics(metrics_results, pairs_file, thresholds)

    experiment.end()

if __name__ == "__main__":
    main()

#def main():
#    model = load_model(model_path)
#    thresholds = np.linspace(0.5, 1.0, num=10)  # A series of thresholds from 0.5 to 1.0

    # Iterate over each pairs file
#    for pairs_file in pairs_files:
#        pairs_file_path = os.path.join(pairs_files_base, pairs_file)

        # Evaluate LFW here and obtain labels (y_true) and similarity scores (y_pred_scores)
#        y_true, y_pred_scores = evaluate_lfw(model, dataset_dir, pairs_file_path)

#        for threshold in thresholds:
#            accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred_scores, threshold)
#            metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

            # Log metrics for each threshold to Comet ML
#            for metric_name, metric_value in metrics.items():
#                experiment.log_metric(f"{pairs_file}_threshold_{threshold}_{metric_name}",
#                                      metric_value,
#                                      step=threshold)

    # End the Comet ML experiment after logging all metrics
#    experiment.end()

