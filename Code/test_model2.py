import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from models.unet import Unet  # Assuming you have a module for your U-Net model
import cv2
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model
model = Unet(in_channels=3, out_channels=2).to(device)
model.load_state_dict(torch.load('./models/unet.pth'))
model.eval()

# Load and preprocess a single image
image_path = 'your_path'  # Replace with the path to your test image
input_image = Image.open(image_path).convert("RGB")

# Define the transformation to resize the image to 256x256 pixels
transform = transforms.Resize((256, 256))

# Apply the transformation to the image
image_redimensionnee = transform(input_image)

# Apply the same transformations as during training
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

input_tensor = preprocess(image_redimensionnee)
input_batch = input_tensor.unsqueeze(0).to(device)

# Perform inference
with torch.no_grad():
    output = model(input_batch)

# Post-process the output
output_np = output.argmax(1).squeeze().cpu().numpy()
segmentation_mask = (output_np * 255).astype(np.uint8)

# Convert the image from RGB to HSV color space
hsv_person = cv2.cvtColor(np.array(image_redimensionnee), cv2.COLOR_RGB2HSV)

# Convert back to RGB color space
person_rgb = cv2.cvtColor(hsv_person, cv2.COLOR_HSV2RGB)

# Convert the segmentation mask to a grayscale image
segmentation_mask_gray = cv2.cvtColor(segmentation_mask, cv2.COLOR_GRAY2BGR)

# Convert the mask to HSV color space
mask_hsv = cv2.cvtColor(segmentation_mask_gray, cv2.COLOR_BGR2HSV)

# Convert back to RGB color space
superposed_mask_rgb = cv2.cvtColor(mask_hsv, cv2.COLOR_HSV2RGB)

# Superimpose the mask on the person's image
superposed_image = hsv_person.copy()

# Initialize variables for averaging pixel values
moy = 0
moy1 = 0
moy2 = 0
m = 0

# Iterate over each pixel of the mask and the person's image
for i in range(mask_hsv.shape[0]):
    for j in range(mask_hsv.shape[1]):
        # Check if the pixel of the mask has a brightness value greater than 0
        if mask_hsv[i, j, 2] > 0:
            # If yes, superimpose the pixel of the mask on the person's image

            moy = moy + superposed_image[i,j,2]%256
            moy1 = moy1 + superposed_image[i,j,1]%256
            moy2 = moy2 + superposed_image[i,j,0]%256
            m += 1
            superposed_image[i, j, 1] += 120 - superposed_image[i, j, 1]  # Change the saturation
            #superposed_image[i, j, 2] += superposed_image[i, j, 2] # Change the brightness
            superposed_image[i, j, 0] = 0 # Adjust the hue of the mask to apply
        # According to Wiki, hue is coded based on the angle on the color wheel:
        # 0° or 180: red; 30: yellow; 60: green; 90: cyan; 120: blue; 150: magenta.

moy = moy / m
moy1 = moy1 / m
moy2 = moy2 / m

print(moy)
print(moy1)
print(moy2)

# Convert the superimposed image back to RGB for display
superposed_image_rgb = cv2.cvtColor(superposed_image, cv2.COLOR_HSV2BGR)

# Save the fused image
cv2.imwrite('Fusion2.png', superposed_image_rgb)

# Display the fused image and segmentation mask
cv2.imshow('Fused Image', superposed_image_rgb)
cv2.imshow('SEG Image', segmentation_mask_gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
