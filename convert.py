import numpy as np
import os
from PIL import Image

def crop_white_padding(image):
    # Convert image to numpy array
    image_np = np.array(image)
    
    # Check the number of channels in the image
    num_channels = image_np.shape[-1]
    
    # Create a mask for non-white pixels
    if num_channels == 4:  # If the image has an alpha channel
        mask = (image_np[:, :, :3] < [255, 255, 255]).any(axis=-1)
    else:  # If the image does not have an alpha channel
        mask = (image_np < [255, 255, 255]).any(axis=-1)
    
    # Find the bounding box of the non-white areas
    coords = np.column_stack(np.where(mask))
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    
    # Crop the image to the bounding box, ensuring no white border is left
    cropped_image = image.crop((x0, y0, x1+1, y1+1))
    return cropped_image


def crop_and_split_image(image):
    # Crop the white padding first
    cropped_img = crop_white_padding(image)
    
    # Split the cropped image into 6 parts horizontally
    width, height = cropped_img.size
    segment_width = width // 6
    segments = []
    for i in range(6):
        left = i * segment_width
        right = (i + 1) * segment_width if (i < 5) else width
        segment = cropped_img.crop((left, 0, right, height))
        segments.append(segment)
    return segments

def process_and_save_images(image_base_path, genres, processed_base_path):
    if not os.path.exists(processed_base_path):
        os.makedirs(processed_base_path)

    for genre in genres:
        genre_path = os.path.join(image_base_path, genre)
        processed_genre_path = os.path.join(processed_base_path, genre)
        
        if not os.path.exists(processed_genre_path):
            os.makedirs(processed_genre_path)
        
        for image_file in os.listdir(genre_path):
            if image_file.endswith('.png'):
                image_path = os.path.join(genre_path, image_file)
                img = Image.open(image_path)
                
                # Process the image to remove white padding and then split it
                segments = crop_and_split_image(img)
                
                # Save each segment
                for idx, segment in enumerate(segments):
                    # Generate a new filename for each segment
                    base_filename, _ = os.path.splitext(image_file)
                    new_filename = f"{base_filename}_part_{idx+1}.png"
                    processed_image_path = os.path.join(processed_genre_path, new_filename)
                    segment.save(processed_image_path)

# def process_and_save_images(image_base_path, genres, processed_base_path):
#     if not os.path.exists(processed_base_path):
#         os.makedirs(processed_base_path)

#     for genre in genres:
#         genre_path = os.path.join(image_base_path, genre)
#         processed_genre_path = os.path.join(processed_base_path, genre)
        
#         if not os.path.exists(processed_genre_path):
#             os.makedirs(processed_genre_path)
        
#         for image_file in os.listdir(genre_path):
#             if image_file.endswith('.png'):
#                 image_path = os.path.join(genre_path, image_file)
#                 img = Image.open(image_path)
                
#                 # Process the image to remove white padding
#                 processed_img = crop_white_padding(img)
                
#                 # Save the processed image
#                 processed_image_path = os.path.join(processed_genre_path, image_file)
#                 processed_img.save(processed_image_path)


path_to_data = '/blue/ruogu.fang/rohanshah1/Data/images_original/'

# Base path where the processed images will be saved
processed_base_path = '/blue/ruogu.fang/rohanshah1/ml/images_processed/'

# Genres list as provided
genres = ["rock", "reggae", "pop", "metal", "jazz", "hiphop", "disco", "country", "classical", "blues"]

# Process images
process_and_save_images(path_to_data, genres, processed_base_path)

# IF YOU COMMENT THE CURRENT PROCESS_AND_SAVE_IMAGES FUNCTION AND UNCOMMENT THE TWO METHODS ABOVE IT, IT WILL DO
#  THE SPLIT INTO SIX IMAGES. THE WAY IT IS NOW JUST REMOVES THE WHITE SPACE
