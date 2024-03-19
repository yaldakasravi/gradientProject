import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

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
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
#from PIL import Image
import matplotlib.pyplot as plt
import random


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

#def load_and_preprocess_image(image_path):
#    img = image.load_img(image_path, target_size=(112, 112))  # Adjust size to 112x112 to match model's expected input
#    img_array = image.img_to_array(img)
    
    #Masking 
    #img_array[ 40:60,30:-30,:] = 0

    #Adding Noise
#    noise_factor = 0.5
#    noise = np.random.random(img_array.shape) * 255 

#    img_array = noise * noise_factor + img_array * (1 - noise_factor)


#    plt.imsave('test_image.png',img_array/255)

#    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
#    return preprocess_input(img_array_expanded_dims)

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
"""
# Example usage
model_path = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/models/GN_W0.5_S2_ArcFace_epoch16.h5'
image_path1 = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/images/Anthony_Hopkins_0001.jpg'
image_path2 = '/home/yaldaw/working_dir/yalda/ghostfacenet-ex/images/Anthony_Hopkins_0002.jpg'

compare_faces(model_path, image_path1, image_path2, num_squares, square_size)

