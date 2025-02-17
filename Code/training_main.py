import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.optim import Adam
from models.unet import Unet
from tools.dataloader import get_dataloaders
from tqdm import tqdm
from PIL import Image
import numpy as np

# Purpose: This script trains a U-Net model for image segmentation using PyTorch.

# Set device (MPS for Apple, CUDA for NVIDIA, or CPU as fallback)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Model checkpoint path
path = "./models/unet.pth"

# Define loss function (Binary Cross-Entropy Loss)
loss_fn = nn.CrossEntropyLoss()

# TensorBoard writer for logging
writer = SummaryWriter()

# Training flag
train = 1

if train:
    # Load training and validation datasets
    train_loader, val_loader = get_dataloaders(
        "dataset/Celeb/train", batch_size=10, valid_size=0.3
    )
    print(len(train_loader))
    print(len(val_loader))

    # Initialize the U-Net model
    model = Unet(in_channels=3, out_channels=2).to(device)

    # Define optimizer (Adam)
    optimizer = Adam(model.parameters(), lr=1e-4)

    # Number of training epochs
    num_epochs = 100

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        
        # Forward propagation on training set
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = loss_fn(outputs, labels)
            writer.add_scalar("Loss/train", loss, epoch)
            print("Training loss : {}".format(loss))
            
            # Backward propagation
            loss.backward()
            
            # Update model parameters
            optimizer.step()

        writer.flush()
        
        # Evaluate model on validation set
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            i = 0
            
            for images, labels in tqdm(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Compute validation loss
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                
                # Compute validation accuracy
                _, preds = torch.max(outputs, 1)
                val_acc += torch.mean((preds == labels.data).float())
                
                # Save sample results
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
    
    # Save trained model
    torch.save(model.state_dict(), "models/unet.pth")
