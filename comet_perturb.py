import numpy as np
import os
from comet_ml import Experiment
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
#from PIL import Image
# Assuming the Comet ML experiment is correctly initialized
experiment = Experiment(api_key="UuHTEgYku8q9Ww3n13pSEgC8d", project_name="inoise-threshold-analysis", workspace="enhancing-gradient")

# Define paths
model_path = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/models/GN_W0.5_S2_ArcFace_epoch16.h5'
dataset_dir = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files_base = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files = [f'pairs_{i:02}.txt' for i in range(1, 11)]  # Adjust the range as needed

def load_and_preprocess_image(image_path, noise_factor):
    img = image.load_img(image_path, target_size=(112, 112))  # Load the image with the target size
    img_array = image.img_to_array(img)  # Convert to a NumPy array

    # Generate Gaussian noise
    noise = np.random.random(size=img_array.shape) * 255

    # Add the noise to the image array directly as float
    img_array = (1 - noise_factor)* img_array + (noise_factor) * noise

    # Ensure the values are still in the valid range
    img_array = np.clip(img_array, 0, 255)

    # Preprocess the image for the model as float
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_input(img_array_expanded_dims.astype(np.float32))

    return processed_img

def get_embedding(model, image_path, noise_factor):
    processed_image = load_and_preprocess_image(image_path, noise_factor)
    return model.predict(processed_image)

def get_cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1.flatten(), embedding2.flatten())

def evaluate_lfw(model, dataset_dir, pairs_file_path, noise_factor):
    similarities, labels = [], []
    with open(pairs_file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        for i in range(0, len(lines), 4):
            img1_path_same = os.path.join(dataset_dir, lines[i])
            img2_path_same = os.path.join(dataset_dir, lines[i+1])
            embedding1_same = get_embedding(model, img1_path_same, noise_factor)
            embedding2_same = get_embedding(model, img2_path_same, noise_factor)
            similarity_same = get_cosine_similarity(embedding1_same, embedding2_same)
            similarities.append(similarity_same)
            labels.append(1)

            img1_path_diff = os.path.join(dataset_dir, lines[i+2])
            img2_path_diff = os.path.join(dataset_dir, lines[i+3])
            embedding1_diff = get_embedding(model, img1_path_diff, noise_factor)
            embedding2_diff = get_embedding(model, img2_path_diff, noise_factor)
            similarity_diff = get_cosine_similarity(embedding1_diff, embedding2_diff)
            similarities.append(similarity_diff)
            labels.append(0)

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

#def main():
#    model = load_model(model_path)
#    noise_levels = [0.0, 0.3, 0.5, 0.6, 0.7]  # Adjusted noise levels
#    thresholds = np.linspace(0.5, 0.7, num=5)

#    for noise_factor in noise_levels:
#        for pairs_file in pairs_files:
#            pairs_file_path = os.path.join(pairs_files_base, pairs_file)
#            y_true, y_pred_scores = evaluate_lfw(model, dataset_dir, pairs_file_path, noise_factor=noise_factor)

#            for threshold in thresholds:
#                accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred_scores, threshold)
#                metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

                # Log metrics for each threshold to Comet ML
#                for metric_name, metric_value in metrics.items():
#                    experiment.log_metric(f"noise_{noise_factor}_{metric_name}_threshold_{threshold}", 
#                                          metric_value, 
#                                          step=threshold)

#    experiment.end()
# for plotting :

def main():
    model = load_model(model_path)
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    avg_metrics = {noise: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for noise in noise_levels}

    for th in np.linspace(0.3,1,num=14):
        for noise_factor in noise_levels:
            all_metrics = []

            pair_acc = []
            pair_precision = []
            pair_recall = []
            pair_f1 = []

            for pairs_file in pairs_files:
                pairs_file_path = os.path.join(pairs_files_base, pairs_file)
                y_true, y_pred_scores = evaluate_lfw(model, dataset_dir, pairs_file_path, noise_factor=noise_factor)
                metrics = calculate_metrics(y_true, y_pred_scores,th)  # ensure this returns a dict
                all_metrics.append(metrics)

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
                #print(f"Summary for {pairs_file} with {noise_factor} masks:")
                #print(f"  Accuracy: {accuracy:.4f}")
                #print(f"  Precision: {precision:.4f}")
                #print(f"  Recall: {recall:.4f}")
                #print(f"  F1 Score: {f1:.4f}\n")
                #print(f"  Threshold: {th}\n")

            print(f"\nSummary for average with {noise_factor} masks and {th} threshold:")
            print(f"  Accuracy: {mean(pair_acc):.4f}")
            print(f"  Precision: {mean(pair_precision):.4f}")
            print(f"  Recall: {mean(pair_recall):.4f}")
            print(f"  F1 Score: {mean(pair_f1):.4f}\n")
        # Calculate average metrics for this noise level
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                metric_values = [m[metric] for m in all_metrics]
                avg_metrics[noise_factor][metric] = np.mean(metric_values)

        save_directory = "threshold-nois_experiment_dump_plot"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
            plt.figure()
            noise_factors = list(noise_levels)
            metric_values = [avg_metrics[noise][metric_name] for noise in noise_levels]
            plt.plot(noise_factors, metric_values, marker='o', linestyle='-')
            plt.title(f"{metric_name.capitalize()} vs. Noise Level")
            plt.xlabel("Noise Level")
            plt.ylabel(metric_name.capitalize())
            plt.grid(True)
            plt.savefig(os.path.join(save_directory, f"{metric_name}_vs_noise_level_at_{th:.2f}_threshold.png"))
            plt.close()

if __name__ == "__main__":
    main()

