import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
import time
import os


#--- Configuration ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 25 #increased from 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = './data'

print(f"[{time.strftime('%H:%M:%S')}] Using device: {DEVICE}")


#--- Data Preparation ---
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30), #Increased from 15
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), #Random Color Adjustments
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5), #Random Tilt
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Load Dataset
dataset_train = torchvision.datasets.OxfordIIITPet(
    root=DATA_DIR,
    split='trainval',
    target_types='category',
    download=False,
    transform=train_transforms
)
dataset_val = torchvision.datasets.OxfordIIITPet(
    root=DATA_DIR,
    split='test',
    target_types='category',
    download=False,
    transform=val_transforms
)

indices = torch.randperm(len(dataset_train)).tolist()
split = int(0.8 * len(dataset_train))
train_data = torch.utils.data.Subset(dataset_train, indices[:split])
val_data = torch.utils.data.Subset(dataset_train, indices[split:])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

print(f"[{time.strftime('%H:%M:%S')}] Data loaded. Train: {len(train_data)}, Val: {len(val_data)}")

#--- Model Setup ---
print(f"[{time.strftime('%H:%M:%S')}] Downloading Pre-trained ResNet18...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = True

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 37)
model = model.to(DEVICE)

#--- Setup Training Tools ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # Added weight decay for regularization

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # Learning rate decay

#--- Training Loop ---
print(f"[{time.strftime('%H:%M:%S')}] Starting Training for {EPOCHS} Epochs...")

best_acc = 0.0

for epoch in range(EPOCHS):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs) #Forward Pass
        loss = criterion(outputs, labels)
        loss.backward() # Backward Pass
        optimizer.step()
        running_loss += loss.item() * inputs.size(0) # Accumulate loss
        _, predicted = torch.max(outputs, 1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()

    scheduler.step() #Update learning rate
    epoch_loss = running_loss / len(train_data)
    epoch_acc = correct_preds / total_preds

    model.eval() # Validation Phase
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
    val_acc = val_correct / len(val_data)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), './models/pet_classifier_best.pth')

    end_time = time.time()
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Loss: {epoch_loss:.4f} | "
          f"Train Acc: {epoch_acc:.2%} | "
          f"Val Acc: {val_acc:.2%} | "
          f"Best: {best_acc:.2%} | "
          f"Time: {end_time - start_time:.1f}s")

print(f"[{time.strftime('%H:%M:%S')}] Training Complete. Best Validation Accuracy: {best_acc:.2%}.")