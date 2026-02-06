import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
import time
import os

#ADV Configs
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = './data'

print(f"[{time.strftime('%H:%M:%S')}] System initialized on {DEVICE}")

train_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
])

val_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def train_adapter(img, target):
    return train_pipeline(img), target
def val_adapter(img, target):
    return val_pipeline(img), target

dataset_train = torchvision.datasets.OxfordIIITPet(root=DATA_DIR, split='trainval', target_types='category', download=False, transforms=train_adapter)
dataset_val = torchvision.datasets.OxfordIIITPet(root=DATA_DIR, split='trainval', target_types='category', download=False, transforms=val_adapter)

indices = torch.randperm(len(dataset_train)).tolist()
split = int(0.8 * len(dataset_train))
train_data = torch.utils.data.Subset(dataset_train, indices[:split])
val_data = torch.utils.data.Subset(dataset_val, indices[split:])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

print(f"[{time.strftime('%H:%M:%S')}] Data loaded. Train: {len(train_data)}, Val: {len(val_data)}")

print(f"[{time.strftime('%H:%M:%S')}] Building EfficientNet-B0...")
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = True

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 37)

model = model.to(DEVICE)

#Label Smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3)

# ------ Training Loop ------
print(f"[{time.strftime('%H:%M:%S')}] Starting PRO Mode Training for {EPOCHS} Epochs...")
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
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_data)
    epoch_acc = correct_preds / total_preds

    model.eval()
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
    val_acc = val_correct / len(val_data)

    scheduler.step(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), './models/pet_classifier_pro.pth')

    end_time = time.time()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Train: {epoch_acc:.2%} | Val: {val_acc:.2%} | Best: {best_acc:.2%} | Time: {end_time - start_time:.1f}s")
print(f"[{time.strftime('%H:%M:%S')}] DONE. Best Pro Model Saved.")