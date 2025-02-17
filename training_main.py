import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.optim import Adam
from models.unet import Unet
from tools.dataloader import get_dataloaders
from tqdm import tqdm
from PIL import Image
import numpy as np


# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Modèle
path = "./models/unet.pth"

# Définir la fonction de perte (ici, la binary cross-entropy)
loss_fn = nn.CrossEntropyLoss()

writer = SummaryWriter()

train = 1

if train:
    train_loader, val_loader = get_dataloaders(
        "dataset/Celeb/train", batch_size=10, valid_size=0.3
    )
    print(len(train_loader))
    print(len(val_loader))

    # Initialiser le modèle U-Net
    model = Unet(in_channels=3, out_channels=2).to(device)

    # Définir l'optimiseur (Adam)
    optimizer = Adam(model.parameters(), lr=1e-4)

    # Nombre d'époques d'entraînement
    num_epochs = 100

    # Boucle d'entraînement
    for epoch in range(num_epochs):
        model.train()
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        # Boucle de propagation avant sur l'ensemble d'entraînement
        for images, labels in tqdm(train_loader):
            # Mettre les données sur le device
            images = images.to(device)
            labels = labels.to(device)

            # Réinitialiser les gradients
            optimizer.zero_grad()

            # Propagation avant
            outputs = model(images)

            # print(labels.shape)
            # print(outputs.shape)

            loss = loss_fn(outputs, labels)
            writer.add_scalar("Loss/train", loss, epoch)
            print("Training loss : {}".format(loss))

            # Propagation arrière
            loss.backward()

            # Mettre à jour les paramètres
            optimizer.step()

        writer.flush()

        # Evaluation du modèle sur l'ensemble de validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            i = 0
            for images, labels in tqdm(val_loader):
                # Mettre les données sur le device
                images = images.to(device)
                labels = labels.to(device)

                # Propagation avant
                outputs = model(images)

                # Calculer la perte
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                # Calculer la précision
                _, preds = torch.max(outputs, 1)

                val_acc += torch.mean((preds == labels.data).float())

                for ii in range(0, images.shape[0]):
                    img = images[ii].detach().cpu().numpy()
                    img = img.transpose((1, 2, 0))
                    img = ((img * 0.5 + 0.5) * 255).astype(np.uint8)
                    img = Image.fromarray(img)
                    img.save("res/res{}.png".format(i))

                    label = labels[ii].detach().cpu().numpy()
                    label = (label * 255).astype(np.uint8)
                    label = Image.fromarray(label)
                    label.save("res/res{}_gt.png".format(i))

                    label = preds[ii].detach().cpu().numpy()
                    label = (label * 255).astype(np.uint8)
                    label = Image.fromarray(label)
                    label.save("res/res{}_pred.png".format(i))

                    i = i + 1

            val_loss /= len(val_loader)
            val_acc = val_acc.float() / len(val_loader)
            print(f"Validation loss: {val_loss:.4f}, Validation acc: {val_acc:.4f}")

    torch.save(model.state_dict(), "models/unet.pth")
