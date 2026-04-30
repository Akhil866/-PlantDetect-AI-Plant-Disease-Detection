import os
import shutil
import random
import copy

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image       

# ============================================================
# Hyperparameters & Configuration
# ============================================================
BATCH_SIZE = 32
NUM_WORKERS = 0        # Set to 4+ on Linux/Mac for faster loading
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
PATIENCE = 2           # Early stopping patience
DROPOUT_RATE = 0.5
SEED = 42


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed):
    """Set seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

# Use script directory as base so relative paths always resolve correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Custom function to safely load images and handle errors
def safe_loader(path):
    """Load an image safely, skipping corrupted files."""
    try:
        # First pass: verify image integrity (consumes the file handle)
        with Image.open(path) as img:
            img.verify()
        # Second pass: actually load the pixel data after verification
        img = Image.open(path)
        img = img.convert("RGB")
        return img
    except (OSError, IOError, SyntaxError) as e:
        print(f"Skipping corrupted image: {path} ({e})")
        return None


# Set paths for your train and valid directories
train_dir = os.path.join(BASE_DIR, 'PlantVillage', 'train')
valid_dir = os.path.join(BASE_DIR, 'PlantVillage', 'valid')
split_done_flag = os.path.join(BASE_DIR, 'PlantVillage', '.split_done')

# Split data if validation directory does not exist
# Uses a flag file to prevent partial re-splits after a crash
if not os.path.exists(split_done_flag):
    print("Valid directory does not exist or split incomplete. Splitting data...")

    # Clean up any partial previous split
    if os.path.exists(valid_dir):
        shutil.rmtree(valid_dir)

    os.makedirs(valid_dir, exist_ok=True)
    class_names = os.listdir(train_dir)

    for class_name in class_names:
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            train_images, valid_images = train_test_split(images, test_size=0.2, random_state=SEED)
            os.makedirs(os.path.join(valid_dir, class_name), exist_ok=True)

            for image in valid_images:
                # Move (not copy) to prevent data leakage between train and validation sets
                shutil.move(
                    os.path.join(class_path, image),
                    os.path.join(valid_dir, class_name, image)
                )

    # Write flag file to mark split as complete
    with open(split_done_flag, 'w') as f:
        f.write('split complete')
    print("Data split complete.")

# Define transforms with data augmentation and normalization
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),           # Randomly flip images
    transforms.RandomRotation(10),               # Randomly rotate images
    transforms.ColorJitter(brightness=0.2),      # Randomly change brightness
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Load datasets with SafeImageFolder class
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = safe_loader(path)
        if img is None:
            return self.__getitem__((index + 1) % len(self.samples))
        return self.transform(img) if self.transform else img, target


train_dataset = SafeImageFolder(train_dir, transform=train_transform)
valid_dataset = SafeImageFolder(valid_dir, transform=valid_transform)

# Wrap training in __main__ guard (required for multiprocessing on Windows)
if __name__ == '__main__':
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # Define the model (ResNet18 with Dropout)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_classes = len(train_dataset.classes)
    model.fc = nn.Sequential(
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(model.fc.in_features, num_classes)
    )

    # Move the model to the device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Using device: {device}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Define loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # ReduceLROnPlateau adapts LR based on validation loss instead of a fixed schedule
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)

    # Mixed precision training (speeds up GPU training significantly)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Early Stopping parameters
    best_valid_loss = float('inf')
    no_improvement_epochs = 0
    best_model_weights = copy.deepcopy(model.state_dict())

    # Training loop with early stopping and mixed precision
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Scaled backward pass for mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # Track training accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation loop
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                running_loss += loss.item()

                # Track validation accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        valid_loss = running_loss / len(valid_loader)
        valid_acc = 100.0 * correct / total
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%, '
              f'LR: {current_lr:.6f}')

        # Learning rate scheduler step (based on validation loss)
        scheduler.step(valid_loss)

        # Early Stopping check
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            no_improvement_epochs = 0  # Reset counter
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= PATIENCE:
                print("Early stopping triggered.")
                break

    # Load the best model weights (from before early stopping)
    model.load_state_dict(best_model_weights)

    # Plot the training and validation loss + accuracy curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    ax1.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training and Validation Loss')

    ax2.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
    ax2.plot(range(1, len(valid_accuracies) + 1), valid_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.set_title('Training and Validation Accuracy')

    plt.tight_layout()

    # Save the trained model FIRST (before plt.show which can block)
    model_path = os.path.join(BASE_DIR, 'PlantVillage_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': train_dataset.classes,
        'num_classes': num_classes,
    }, model_path)
    print(f"Model training complete and saved to {model_path}")

    # Save plot to file, then show interactively
    plot_path = os.path.join(BASE_DIR, 'training_curves.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Training curves saved to {plot_path}")
    plt.show()
