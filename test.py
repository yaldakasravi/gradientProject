import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import os
from scipy.spatial.distance import cosine
#from PIL import Image
"""
#for swapping same person 
def load_and_preprocess_image(image_path, swap_image_path, mask_thickness):
    # Load the target image
    img = image.load_img(image_path, target_size=(112, 112))
    img_array = image.img_to_array(img)

    # Load the image to swap with
    swap_img = image.load_img(swap_image_path, target_size=(112, 112))
    swap_img_array = image.img_to_array(swap_img)

    # Define the number of rows to swap based on the mask thickness
    num_rows_to_swap = int(112 * mask_thickness)

    # Swap a horizontal strip across the center of the image
    start_row = (112 - num_rows_to_swap) // 2
    end_row = start_row + num_rows_to_swap
    img_array[start_row:end_row, :, :] = swap_img_array[start_row:end_row, :, :]
    plt.imsave('test_image.png', img_array.astype(np.uint8))
    # Preprocess the image for the model
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)

def get_embedding(model, image_path, swap_image_path, mask_thickness):
    processed_image = load_and_preprocess_image(image_path, swap_image_path, mask_thickness)
    return model.predict(processed_image)

def compare_faces(model_path, image_path1, image_path2, mask_thickness):
    model = load_model(model_path)
    
    # Get the embeddings after the swap has been done
    embedding1 = get_embedding(model, image_path1, image_path2, mask_thickness)
    embedding2 = get_embedding(model, image_path2, image_path1, mask_thickness)
    
    # Compute similarity (1 - cosine distance)
    similarity = 1 - cosine(embedding1.flatten(), embedding2.flatten())
    print(f"Similarity: {similarity}")
    
    # Decision threshold can be adjusted based on model and requirements
    if similarity > 0.5:  
        print("The images are of the same person.")
    else:
        print("The images are of different people.")


"""
#for random masking
"""
num_squares = 10
square_size = 20

def apply_random_mask(img_array, num_squares, square_size):
    h, w, _ = img_array.shape
    for _ in range(num_squares):
        x1 = random.randint(0, w - square_size)
        y1 = random.randint(0, h - square_size)
        img_array[y1:y1 + square_size, x1:x1 + square_size, :] = 0
    return img_array

def load_and_preprocess_image(image_path, num_squares, square_size):
    img = image.load_img(image_path, target_size=(112, 112))
    img_array = image.img_to_array(img)

    # Apply random masking if specified
    if num_squares > 0:
        img_array = apply_random_mask(img_array, num_squares, square_size)

    # Save the masked image for inspection
    plt.imsave('test_image.png', img_array.astype(np.uint8))

    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)

def get_embedding(model, image_path, num_squares, square_size):
    processed_image = load_and_preprocess_image(image_path, num_squares, square_size)
    return model.predict(processed_image)

def compare_faces(model_path, image_path1, image_path2, num_squares, square_size):
    model = load_model(model_path)

    embedding1 = get_embedding(model, image_path1, num_squares, square_size)
    embedding2 = get_embedding(model, image_path2, num_squares, square_size)

    # Compute similarity
    similarity = cosine_similarity(embedding1, embedding2)
    print(f"Similarity: {similarity[0][0]}")

    # Decision threshold can be adjusted based on model and requirements
    if similarity[0][0] > 0.5:
        print("The images are of the same person.")
    else:
        print("The images are of different people.")

"""
"""
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(112, 112))  # Load the image with the target size
    img_array = image.img_to_array(img)  # Convert to a NumPy array

    # Define the region of interest where noise will be added
    mask_region = img_array[40:60, 30:-30, :]
    
    # Define the standard deviation of the Gaussian noise
    noise_sigma = 25  # This controls how "noisy" the image will be

    # Generate Gaussian noise
    noise = np.random.normal(loc=0.0, scale=noise_sigma, size=mask_region.shape)
    
    # Since img_array is likely to be uint8, noise also needs to be in the same type
    noise = noise.astype(np.uint8)
    
    # Add the noise to the specified part of the image array
    img_array[40:60, 30:-30, :] = mask_region + noise

    # Ensure the values are still in the range [0, 255]
    img_array = np.clip(img_array, 0, 255)
    
    # Save the noisy image for inspection
    plt.imsave('test_image.png', img_array.astype(np.uint8))

    # Preprocess the image for the model
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_input(img_array_expanded_dims.astype(np.float32))

    return processed_img

def load_and_preprocess_image(image_path):
    
    img = image.load_img(image_path, target_size=(112, 112))  # Adjust size to 112x112 to match model's expected input
    img_array = image.img_to_array(img)

    # Create a mask with the same dimensions as the image, filled with zeros (black)
    mask = np.zeros_like(img_array)

    # Define the mask region (here we want to preserve the region and make everything else black)
    # Set the mask region to 1 (white)
    mask[40:60, 30:-30, :] = 1

    # Apply the mask to the image array
    img_array = img_array * mask
"""
def load_and_preprocess_image(image_path):
    #Adding Noise
    img = image.load_img(image_path, target_size=(112, 112))  # Adjust size to match model's expected input
    img_array = image.img_to_array(img)
    
    # Create a noise array with the same dimensions as the image
    noise = np.random.rand(*img_array.shape) * 255

    # Create a mask for the region you want to preserve (keeping it as before)
    mask = np.zeros_like(img_array, dtype=bool)
    mask[40:60, 30:-30, :] = True
    
    # Create an inverse mask for applying noise (where mask is False)
    inverse_mask = ~mask
    
    # Apply noise only to the regions outside the mask region
    img_array[inverse_mask] = noise[inverse_mask]

    plt.imsave('test_image.png',img_array/255)

    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)

def get_embedding(model, image_path):
    processed_image = load_and_preprocess_image(image_path)
    return model.predict(processed_image)

def compare_faces(model_path, image_path1, image_path2):
    model = load_model(model_path)
    
    embedding1 = get_embedding(model, image_path1)
    embedding2 = get_embedding(model, image_path2)
    
    # Compute similarity
    similarity = cosine_similarity(embedding1, embedding2)
    print(f"Similarity: {similarity[0][0]}")
    
    # Decision threshold can be adjusted based on model and requirements
    if similarity > 0.5:  
        print("The images are of the same person.")
    else:
        print("The images are of different people.")


# Example usage
model_path = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/models/GN_W0.5_S2_ArcFace_epoch16.h5'
image_path1 = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/images/Anthony_Hopkins_0001.jpg'
image_path2 = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/images/Anthony_Hopkins_0002.jpg'
mask_thickness = 0.5  # Set the mask thickness as required

#compare_faces(model_path, image_path1, image_path2, mask_thickness)
compare_faces(model_path, image_path1, image_path2)

