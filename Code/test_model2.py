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
image_path = 'your_path'  # Replace with the path to your t²est image
input_image = Image.open(image_path).convert("RGB")

# Définir la transformation pour redimensionner l'image en 256x256 pixels
transform = transforms.Resize((256, 256))

# Appliquer la transformation à l'image
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

# Convertir l'image de RGB à HSV
hsv_person = cv2.cvtColor(np.array(image_redimensionnee), cv2.COLOR_RGB2HSV)

# Repasser dans l'espace RGB
person_rgb = cv2.cvtColor(hsv_person, cv2.COLOR_HSV2RGB)

# Convertir le masque en une image en niveaux de gris
segmentation_mask_gray = cv2.cvtColor(segmentation_mask, cv2.COLOR_GRAY2BGR)


# Convertir le masque en une image avec trois canaux (HSV)
mask_hsv = cv2.cvtColor(segmentation_mask_gray, cv2.COLOR_BGR2HSV)

# Repasser dans l'espace RGB
superposed_mask_rgb = cv2.cvtColor(mask_hsv, cv2.COLOR_HSV2RGB)

# Superposer le masque sur l'image de la personne
superposed_image = hsv_person.copy()


moy = 0
moy1 = 0
moy2 = 0
m = 0
# Parcourir chaque pixel du masque et de l'image de la personne
for i in range(mask_hsv.shape[0]):
    for j in range(mask_hsv.shape[1]):
        # Vérifier si le pixel du masque a une valeur de luminosité supérieure à 0
        if mask_hsv[i, j, 2] > 0:
            # Si oui, superposer le pixel du masque sur l'image de la personne

            moy = moy + superposed_image[i,j,2]%256
            moy1 = moy1 + superposed_image[i,j,1]%256
            moy2 = moy2 + superposed_image[i,j,0]%256
            m +=1
            superposed_image[i, j, 1] += 120 - superposed_image[i, j, 1]  # Saturation que l'on change
            #superposed_image[i, j, 2] += superposed_image[i, j, 2] # luminosité que l'on change
            superposed_image[i, j, 0] = 0 #Ajuster ici la couleur (la teinte) du mask que l'on veut appliquer
        # D'apres Wiki, La teinte est codée suivant l'angle qui lui correspond sur le cercle des couleurs :

#0° ou 180 : rouge ;
#30 : jaune ;
#60 : vert ;
#90 : cyan ;
#120 : bleu ;
#150 : magenta.

moy = moy/m
moy1 = moy1/m
moy2 = moy2/m

print(moy)
print(moy1)
print(moy2)


# Repasser dans l'espace RGB pour faire afficher l'image
superposed_image_rgb = cv2.cvtColor(superposed_image, cv2.COLOR_HSV2BGR)

# Enregistrer l'image
cv2.imwrite('Fusion2.png', superposed_image_rgb)

# Afficher l'image fusionnée
cv2.imshow('Fused Image', superposed_image_rgb)
cv2.imshow('SEG Image', segmentation_mask_gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
