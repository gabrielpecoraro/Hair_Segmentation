import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from models.unet import Unet  # Assuming you have a module for your U-Net model
import cv

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")
# Load pre-trained model
model = Unet(in_channels=3, out_channels=2).to(device)
model.load_state_dict(torch.load("./models/unet.pth"))
model.eval()

# Load and preprocess a single image
image_path = "your_path"  # Replace with the path to your test image
input_image = cv2.imread(image_path)

hsv_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)


# Define transformation to resize at 256x256
transform = transforms.Resize((256, 256))

# Apply transformation
image_redimensionnee = transform(input_image)

# Apply the same transformations as during training
preprocess = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

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
image_redimensionnee.convert("RGB")


# Show the predicted segmentation mask
result_image = Image.fromarray(segmentation_mask)
result_image.save("Predicted_your_file")

# Load image with grey scale
image_gris = Image.open("Predicted_your_file")

largeur, hauteur = image_gris.size

# convert in RGB
# image_rgb = image_gris.convert('RGB')

couleur_fond = (0, 0, 0)  # (R, G, B)

nouvelle_image = Image.new("RGB", (largeur, hauteur), couleur_fond)

ave = 0
i = 0
for y in range(hauteur):
    for x in range(1, largeur):
        # Retrive grey pixel value
        valeur_gris = image_gris.getpixel((x, y))
        valeur_gris_precedente = image_gris.getpixel((x - 1, y))

        valeur_originale = image_redimensionnee.getpixel((x, y))
        valeur_originale_precedente = image_redimensionnee.getpixel((x - 1, y))

        r1, v1, b1 = valeur_originale
        r2, v2, b2 = valeur_originale_precedente

        if valeur_gris > 0:
            moy = (r1 - r2) ** 2 + (v1 - v2) ** 2 + (b1 - b2) ** 2
            if moy < 20:
                ave += int((r1 + v1 + b1) / 3)
                i += 1
                print("moyenne pixels blancs =", moy)

        # Update with new value
        # nouvelle_image.putpixel((x, y), couleur)

print("nb_pixel =", i)
ave = int(ave / i)
print("moyenne total pixels blancs =", ave)

couleur = (int(valeur_gris / 2), 0, int(valeur_gris / 2))  # Rouge ,  vert , sans bleu

for y in range(hauteur):
    for x in range(largeur):
        # Récupérer la valeur de gris du pixel
        valeur_gris = image_gris.getpixel((x, y))

        valeur_originale = image_redimensionnee.getpixel((x, y))

        if valeur_gris > 0:
            if ave <= 50:
                alpha = 0.10
                beta = 0.9
                # Créer une couleur  en fonction de la valeur de gris
                couleur = (int(valeur_gris), 0, 0)  # Rouge ,  vert , sans bleu
                valeur = tuple(
                    int(alpha * i + beta * j) for i, j in zip(couleur, valeur_originale)
                )

                nouvelle_image.putpixel((x, y), valeur)
            if (50 < ave) and (ave <= 100):
                alpha = 0.2
                beta = 0.95
                # Créer une couleur  en fonction de la valeur de gris
                couleur = (int(valeur_gris), 0, 0)  # Rouge ,  vert , sans bleu
                valeur = tuple(
                    int(alpha * i + beta * j) for i, j in zip(couleur, valeur_originale)
                )
                nouvelle_image.putpixel((x, y), valeur)
            if ave > 100:
                alpha = 0.15
                beta = 0.95
                # Créer une couleur  en fonction de la valeur de gris
                couleur = (int(valeur_gris), 0, 0)  # Rouge ,  vert , sans bleu
                valeur = tuple(
                    int(alpha * i + beta * j) for i, j in zip(couleur, valeur_originale)
                )
                nouvelle_image.putpixel((x, y), valeur)
        else:
            nouvelle_image.putpixel((x, y), valeur_originale)


# Save image
# image_rgb.save('image_rouge.png')
image_redimensionnee.save("your_file")
nouvelle_image.save("Predicted_your_file")

# open image
image_initiale = Image.open("your_file")

# open red mask
masque_rouge = Image.open("Predicted_your_file")


# Superpose images
# image_superposee = Image.alpha_composite(image_initiale.convert("RGBA"), masque_rouge.convert("RGBA"))

# save new image
# image_superposee.save("image_superposee2.png")

# display image
nouvelle_image.show("image avec mask")

# image_superposee.show('image superpose')

result_image.show(title="Predicted Segmentation Mask")

# Save the result image
result_image.save("result.png")
