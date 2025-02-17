from PIL import Image
import os

# Purpose: This script converts all images in the input folder to RGB format and saves them in the output folder.
def convert_to_RGB(input_folder, output_folder):
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # List all files in the input folder
    files = os.listdir(input_folder)
    
    for file in files:
        # Process only files with specified extensions (sRGB formats and PBM for grayscale compatibility)
        if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.pbm'):
            
            # Construct full input file path
            input_path = os.path.join(input_folder, file)
            
            # Open the image file
            img = Image.open(input_path)
            
            # Convert the image to RGB format
            img_rgb = img.convert('RGB')
            
            # Construct full output file path
            output_path = os.path.join(output_folder, file)
            
            # Save the RGB image
            img_rgb.save(output_path)
            
            print(f"Image {file} successfully converted to RGB format.")

# Define input and output directories for training and testing datasets
input_folder_train = 'res'
output_folder_train = 'res'

input_folder_test = 'dataset/Celeb/test/masks'
output_folder_test = 'dataset/Celeb/test/masks'

# Convert images in training and testing datasets to RGB format
convert_to_RGB(input_folder_train, output_folder_train)
convert_to_RGB(input_folder_test, output_folder_test)
