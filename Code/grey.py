from PIL import Image
import os

# Purpose: This script converts all images in the input folder to grayscale and saves them in the output folder.
def convert_to_gray(input_folder, output_folder):
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
            
            # Convert the image to grayscale
            img_gray = img.convert('L')
            
            # Construct full output file path
            output_path = os.path.join(output_folder, file)
            
            # Save the grayscale image
            img_gray.save(output_path)
            
            print(f"Image {file} successfully converted to grayscale.")

# Define input and output directories for training and testing datasets
input_folder_train = 'dataset/Figaro1k_small/train/masks'
output_folder_train = 'dataset/Figaro1k_small/train/masks'

input_folder_test = 'dataset/Figaro1k_small/test/masks'
output_folder_test = 'dataset/Figaro1k_small/test/masks'

# Convert images in training and testing datasets to grayscale
convert_to_gray(input_folder_train, output_folder_train)
convert_to_gray(input_folder_test, output_folder_test)
