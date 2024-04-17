from utils import read_pairs
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import random

# Parameters
model_path = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/models/GN_W0.5_S2_ArcFace_epoch16.h5'
dataset_dir = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
pairs_files_base = '/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled'
#pairs_files = [f'pairs_{i:02}.txt' for i in range(1, 11)]
#pairs_files = [os.path.join('/home/yaldaw/scratch/yaldaw/dataset/lfw_funneled', f'pairs_{i:02}.txt') for i in range(1, 11)]
pairs_files = [os.path.join(dataset_dir, f'pairs_{i:02}.txt') for i in range(1, 2)]
thresholds = np.linspace(0.0, 1, num=20)
num_squares_range = range(1, 11)
square_size = 20

# Load the model
model = load_model(model_path)

def preprocess_image(image_path):
    #print("Attempting to open image at:", image_path)
    image = Image.open(image_path).resize((112, 112))
    image = np.array(image, dtype='float32')
    image /= 255.0  # Normalization
    return image

def mask_image(image, num_squares, square_size):
    for _ in range(num_squares):
        x = random.randint(0, image.shape[1] - square_size)
        y = random.randint(0, image.shape[0] - square_size)
        image[y:y+square_size, x:x+square_size, :] = 0
    return image

def calculate_similarity(image1, image2):
    # Assuming the model outputs embeddings and cosine similarity is used
    emb1 = model.predict(np.expand_dims(image1, axis=0))
    emb2 = model.predict(np.expand_dims(image2, axis=0))
    similarity = np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity

def main():
    # Adjust results structure to separate each square number under each threshold
    results = {num_squares: {threshold: [] for threshold in thresholds} for num_squares in num_squares_range}
    for num_squares in num_squares_range:
        for threshold in thresholds:
            accuracies = []
            for pairs_file in pairs_files:
                pairs = read_pairs(pairs_file)
                if not pairs:
                    continue
                tp = fp = tn = fn = 0
                for file1, file2, is_same in pairs:
                    image1 = preprocess_image(os.path.join(dataset_dir, file1))
                    image2 = preprocess_image(os.path.join(dataset_dir, file2))
                    image1 = mask_image(np.copy(image1), num_squares, square_size)
                    image2 = mask_image(np.copy(image2), num_squares, square_size)

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
                    accuracies.append(0)  # Append 0 to avoid division by zero
                else:
                    accuracies.append((tp + tn) / total_comparisons)
            results[num_squares][threshold] = np.mean(accuracies)

    # Plotting
    save_directory = "one-threshold-mask-black-bothside_plot"
    os.makedirs(save_directory, exist_ok=True)
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(num_squares_range)))
    for idx, num_squares in enumerate(num_squares_range):
        thresholds_list = list(thresholds)
        accuracies_list = [results[num_squares][threshold] for threshold in thresholds]
        plt.plot(thresholds_list, accuracies_list, 'o-', label=f'Num Squares {num_squares}', color=colors[idx])

    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Effect of Image Masking both images on Face Authentication Accuracy Across Different Thresholds')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_directory, 'accuracy_vs_thresholds_by_num_squares.png')
    plt.savefig(save_path)
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()

