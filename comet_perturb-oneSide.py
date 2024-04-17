from comet_ml import Experiment
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
# Initialize your Comet ML experiment here
experiment = Experiment(api_key="UuHTEgYku8q9Ww3n13pSEgC8d", project_name="full-noise-oneside_effect", workspace="enhancing-gradient")

# Define paths
model_path = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/models/GN_W0.5_S2_ArcFace_epoch16.h5'
dataset_dir = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files_base = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files = [os.path.join(dataset_dir, f'pairs_{i:02}.txt') for i in range(1, 11)]

# Load the model
model = load_model(model_path)

def preprocess_image(image_path):
    image = Image.open(image_path).resize((112, 112))
    image = np.array(image, dtype='float32')
    image /= 255.0  # Normalize
    return image

def add_noise_to_image(image, level):
    # Calculate the noise intensity
    noise_intensity = level * 255  # Scale by 255 as pixel values range from 0 to 255
    
    # Generate noise for the entire image size
    noise = np.random.normal(loc=0.0, scale=noise_intensity, size=image.shape)

    # Add noise to the entire image
    noisy_image = image + noise
    
    # Ensure pixel values remain within [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)

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
                file1 = os.path.join(dataset_dir, lines[i].strip())
                file2 = os.path.join(dataset_dir, lines[i + 1].strip())
                if os.path.isfile(file1) and os.path.isfile(file2):
                    pairs.append((file1, file2, True))
    return pairs

def main():
    thresholds = np.linspace(0.3, 1, num=14)  # Threshold levels
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
                    image1 = add_noise_to_image(image1, noise_level)
                    #image2 = add_noise_to_image(image2, noise_level)

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
    save_directory = "noise-fullmasking-oneside_plot"
    os.makedirs(save_directory, exist_ok=True)
    plt.figure(figsize=(10, 8))
    for noise_level in noise_levels:
        accuracies = [results[threshold][noise_level] for threshold in thresholds]
        plt.plot(thresholds, accuracies, label=f'Noise Level {noise_level:.2f}')

    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Effect of Noise Addition to Eye Regions on Face Recognition Accuracy Across Thresholds')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_directory, 'accuracy_vs_threshold_by_noise_level.png')
    plt.savefig(save_path)
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()

