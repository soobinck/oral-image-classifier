import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2
import torchvision.transforms.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import timm  # EfficientNet models

# Define data directory
data_dir = "Images_classified_abbridged_100"

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
balanced_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
class_names = balanced_dataset.classes
print("Class Names:", class_names)

# Split into train and test sets (80% train, 20% test)
train_size = int(0.8 * len(balanced_dataset))
test_size = len(balanced_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(balanced_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load pre-trained EfficientNet-B3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=len(class_names))

# Add Dropout and L2 Regularization
for param in model.parameters():
    param.requires_grad = True  # Fine-tuning all layers

model.classifier = nn.Sequential(
    nn.Dropout(0.3),  # Prevent overfitting
    nn.Linear(model.classifier.in_features, len(class_names))
)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)  # L2 Regularization
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Early Stopping
class EarlyStopper:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = np.inf
        self.counter = 0

    def should_stop(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

early_stopper = EarlyStopper(patience=7)

# Training loop
num_epochs = 20
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100 * correct / total
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_acc = 100 * correct / total
    val_losses.append(val_loss / len(test_loader))
    val_accuracies.append(val_acc)

    scheduler.step(val_loss / len(test_loader))

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    if early_stopper.should_stop(val_loss / len(test_loader)):
        print("Early stopping triggered. Stopping training.")
        break

# Save model
torch.save(model.state_dict(), "oral_image_classifier_effnet.pth")
print("Model saved as oral_image_classifier_effnet.pth")

# Confusion Matrix
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

conf_matrix = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
# Save the confusion matrix plot
output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

conf_matrix_filepath = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(conf_matrix_filepath)
plt.close()  # Close the plot after saving
print(f"Confusion matrix plot saved to {conf_matrix_filepath}")

# Plot and save training/validation accuracy
plt.figure(figsize=(10, 8))
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy", color='blue')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Validation Accuracy", color='green')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and Validation Accuracy")
accuracy_filepath = os.path.join(output_dir, "accuracy_plot.png")
plt.savefig(accuracy_filepath)
plt.close()  # Close the plot after saving
print(f"Accuracy plot saved to {accuracy_filepath}")
