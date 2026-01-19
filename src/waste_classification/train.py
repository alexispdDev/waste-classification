import glob
import os
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


# Simple preprocessing
input_size = 224

# ImageNet normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

learning_rate = 0.01
size_inner = 100
num_epochs = 50

# Architecture used during training
trained_size_inner = 100      # ← change to whatever you trained with
num_classes        = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()


class WasteDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for label_name in self.classes:
            label_dir = os.path.join(data_dir, label_name)
            for img_name in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, img_name))
                self.labels.append(self.class_to_idx[label_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    

class WasteClassifierMobileNet(nn.Module):
    def __init__(self, size_inner=100, num_classes=6):
        super(WasteClassifierMobileNet, self).__init__()
        
        self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze last block (features[18] is last inverted residual block)
        for param in self.base_model.features[18].parameters():
            param.requires_grad = True
            
        self.base_model.classifier = nn.Identity()
        
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.inner = nn.Linear(1280, size_inner)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(size_inner, num_classes)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.inner(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x
    

def train_and_evaluate(model, optimizer, train_loader, val_loader, criterion, num_epochs, device):
    best_val_accuracy = 0.0  # Initialize variable to track the best validation accuracy

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Checkpoint the model if validation accuracy improved
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            checkpoint_path = f'model/mobilenet_v4_{epoch+1:02d}_{val_acc:.3f}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')


def make_model(
        learning_rate=0.01,
        size_inner=100,
):
    model = WasteClassifierMobileNet(
        num_classes=6,
        size_inner=size_inner,
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer


def extract_accuracy(filename):
    """
    Extracts the accuracy value from filenames like:
    mobilenet_v3A_06_0.845.pth
    """
    match = re.search(r"_([0-9]*\.[0-9]+)\.pth$", filename)
    if match:
        return float(match.group(1))
    return -1  # If something goes wrong

def find_best_checkpoint():
    # Find best checkpoint
    list_of_files = glob.glob('model/mobilenet_v*.pth')
    best_model_file = [x for x in list_of_files if str(max([extract_accuracy(x) for x in list_of_files])) in x][0]
    print(f"Loading model from: {best_model_file}")

    # Load model
    model = WasteClassifierMobileNet(
        size_inner=trained_size_inner, 
        num_classes=num_classes
    )

    model.load_state_dict(torch.load(best_model_file))
    model.to(device)
    model.eval()
    return model


def run():
    # Simple transforms - just resize and normalize
    train_transforms = transforms.Compose([
        transforms.RandomRotation(10),           # Rotate up to 10 degrees
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # Zoom
        transforms.RandomHorizontalFlip(),       # Horizontal flip
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


    # Create dataloaders
    train_dataset = WasteDataset(
        data_dir='data/train',
        transform=train_transforms
    )
    val_dataset = WasteDataset(
        data_dir='data/val',
        transform=val_transforms
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Train the model
    print(f'\n=== Leaning rate: {learning_rate}, size_inner: {size_inner} ===')
    model, optimizer = make_model(learning_rate=learning_rate, size_inner=size_inner)
    train_and_evaluate(model, optimizer, train_loader, val_loader, criterion, num_epochs, device)

    model_to_export = find_best_checkpoint()

    # Export to ONNX
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # Export to ONNX
    onnx_path = "model/waste_classifier_mobilenet_v4.onnx"

    torch.onnx.export(
        model_to_export,
        dummy_input,
        onnx_path,
        verbose=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Model exported to {onnx_path}")


if __name__ == '__main__':
    run()
