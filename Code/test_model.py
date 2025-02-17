import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from models.unet import Unet  # Assuming you have a module for your U-Net model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model
model = Unet(in_channels=3, out_channels=2).to(device)
model.load_state_dict(torch.load('./models/unet.pth'))
model.eval()

# Load and preprocess a single image
image_path = 'python.jpg'  # Replace with the path to your test image
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



# Visualize the original image, ground truth mask, and predicted segmentation mask
image_redimensionnee.show(title="Original Image")
image_redimensionnee.convert('RGB')


# Show the predicted segmentation mask
result_image = Image.fromarray(segmentation_mask)
result_image.save('predicted_gab.jpg')

# Charger l'image en niveaux de gris
image_gris = Image.open('Predicted_gab.jpg')

largeur, hauteur = image_gris.size

# Convertir l'image en mode RGB (couleurs)
#image_rgb = image_gris.convert('RGB')

couleur_fond = (0, 0, 0)  # (R, G, B)

nouvelle_image = Image.new('RGB', (largeur, hauteur), couleur_fond)

alpha = 0.10
beta = 0.90


for y in range(hauteur):
    for x in range(largeur):
        # Récupérer la valeur de gris du pixel
        valeur_gris = image_gris.getpixel((x, y))
        
        valeur_originale = image_redimensionnee.getpixel((x,y))
        
        
        # Créer une couleur rouge en fonction de la valeur de gris
        couleur = ( 0, int(valeur_gris), 0)  # Rouge ,  vert , bleu
        valeur = tuple( int( alpha*x + beta*y )  for x, y in zip(couleur, valeur_originale))

        if valeur_gris > 0 :
           nouvelle_image.putpixel((x, y), valeur)
        else:
            nouvelle_image.putpixel((x, y), valeur_originale)


# Enregistrer l'image résultante
#image_rgb.save('image_rouge.png')
image_redimensionnee.save('python.jpg')
nouvelle_image.save('Predicted_gab.png')

# Ouvrir l'image initiale
image_initiale = Image.open("python.jpg")

# Ouvrir le masque rouge
masque_rouge = Image.open("Predicted_gab.png")


# Superposer le masque rouge redimensionné sur l'image initiale
#image_superposee = Image.alpha_composite(image_initiale.convert("RGBA"), masque_rouge.convert("RGBA"))

# Sauvegarder l'image superposée
#image_superposee.save("image_superposee2.png")

# Afficher l'image
nouvelle_image.show('image avec mask')

#image_superposee.show('image superpose')

result_image.show(title="Predicted Segmentation Mask")

# Save the result image
result_image.save('rouge.png')