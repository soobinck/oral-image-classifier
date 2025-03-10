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
from sklearn.metrics import confusion_matrix, classification_report
import cv2
import torchvision.transforms.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import timm  # EfficientNet models
import json  # To save history as JSON
import pandas as pd  # For saving classification report to CSV
from sklearn.model_selection import StratifiedShuffleSplit

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
#
# # Split into train and test sets (80% train, 20% test)
# train_size = int(0.8 * len(balanced_dataset))
# test_size = len(balanced_dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(balanced_dataset, [train_size, test_size])
#

# Get the labels for the dataset
labels = balanced_dataset.targets

# Create a StratifiedShuffleSplit object
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Split the indices into train and test sets
for train_idx, test_idx in sss.split(np.zeros(len(labels)), labels):
    train_dataset = torch.utils.data.Subset(balanced_dataset, train_idx)
    test_dataset = torch.utils.data.Subset(balanced_dataset, test_idx)

# Print the number of samples in each set
print(f"Training set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")




train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

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

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)


# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0

    def should_stop(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


early_stopper = EarlyStopping(patience=3, delta=0.01)

# Define lists to store history
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Number of epochs
num_epochs = 50

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    # Calculate training accuracy and loss
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation step
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    # Calculate validation accuracy and loss
    val_loss = running_val_loss / len(test_loader)
    val_acc = 100 * correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # Print stats for each epoch
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Early stopping logic
    if early_stopper.should_stop(val_loss):
        print("Early stopping triggered. Stopping training.")
        break

# Save the model
torch.save(model.state_dict(), "oral_image_classifier_effnet.pth")
print("Model saved as oral_image_classifier_effnet.pth")

# Save the training and validation history as a JSON file
history = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "train_accuracies": train_accuracies,
    "val_accuracies": val_accuracies
}

output_history_file = "training_validation_history.json"
with open(output_history_file, 'w') as f:
    json.dump(history, f)

print(f"Training and validation history saved to {output_history_file}")

# Plot and save training/validation loss
plt.figure(figsize=(10, 8))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", color='blue')
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", color='red')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
loss_filepath = os.path.join("output_results", "loss_plot.png")
plt.savefig(loss_filepath)
plt.close()  # Close the plot after saving
print(f"Loss plot saved to {loss_filepath}")

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
sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues",
            cbar=False)
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

# Generate classification report
report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Save the classification report to CSV
report_filepath = os.path.join(output_dir, "classification_report.csv")
report_df.to_csv(report_filepath)

print(f"Classification report saved to {report_filepath}")


def generate_gradcam(model, input_image, target_class, device):
    """
    Generates Grad-CAM visualization for a given input image.
    """

    # Hook function to capture the feature map
    def hook_fn(module, input, output):
        global feature_map
        feature_map = output
        # Retain gradients for feature_map
        feature_map.retain_grad()

    # Get the last convolutional layer in EfficientNet
    last_block = model.blocks[-1]  # The last block in the EfficientNet model
    last_conv_layer = last_block[0]  # The first layer inside the last block, which is a convolution

    # Register hook on the last convolutional layer
    hook = last_conv_layer.register_forward_hook(hook_fn)

    # Send image to device
    input_image = input_image.unsqueeze(0).to(device)

    # Forward pass
    model.zero_grad()
    output = model(input_image)

    # Compute the gradient of the predicted class
    output[:, target_class].backward()

    # Get the gradients of the feature map
    gradients = feature_map.grad

    # Ensure that gradients are correctly captured
    if gradients is None:
        raise ValueError("Gradients were not computed. Ensure the backward pass was done correctly.")

    # Pool the gradients across all the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Weight the channels by the gradients
    for i in range(feature_map.shape[1]):
        feature_map[:, i, :, :] *= pooled_gradients[i]

    # Create heatmap
    heatmap = torch.mean(feature_map, dim=1).squeeze()

    # Normalize the heatmap
    heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
    heatmap = cv2.resize(heatmap, (input_image.shape[3], input_image.shape[2]))
    heatmap = heatmap / heatmap.max()

    # Remove hook
    hook.remove()

    return heatmap


def save_gradcam_images(model, data_loader, class_names, device, output_dir='gradcam'):
    """
    Generate Grad-CAM for all images in the dataset and save them.
    """
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Loop through the dataset
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # Generate and save Grad-CAM images
        for j in range(len(images)):
            image = images[j].cpu().detach()
            label = labels[j].item()
            pred = predicted[j].item()

            # Generate Grad-CAM heatmap for the image
            heatmap = generate_gradcam(model, image, pred, device)

            # Convert the image tensor to a numpy array
            img = images[j].cpu().detach().numpy().transpose(1, 2, 0)
            img = np.uint8(img * 255)  # Convert to 8-bit image

            # Superimpose the heatmap on the original image
            heatmap_resized = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(img, 0.6, heatmap_resized, 0.4, 0)

            # Save the Grad-CAM image
            image_name = f"{i * len(images) + j}.png"
            file_name = f"{image_name}-{class_names[label]}-{class_names[pred]}.png"
            save_path = os.path.join(output_dir, file_name)

            cv2.imwrite(save_path, superimposed_img)
            print(f"Grad-CAM image saved at {save_path}")


# Generate and save Grad-CAM for all training images
save_gradcam_images(model, train_loader, class_names, device, output_dir='gradcam/train')

# Generate and save Grad-CAM for all test images
save_gradcam_images(model, test_loader, class_names, device, output_dir='gradcam/test')
