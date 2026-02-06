import torchvision
from torchvision import transforms
import os

print("[1/3] Setting up directories...")
data_dir = './data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print(" [2/3] Downloading the Oxford-IIIT Pet Dataset...")
simple_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

try:
    dataset = torchvision.datasets.OxfordIIITPet(
        root = data_dir,
        split = 'trainval',
        target_types = 'category',
        download = True,
        transform = simple_transform
    )
    print(f" Download Complete! ")
    print(f" Dataset size: {len(dataset)} images. ")

    img, label = dataset[0]
    print(f" Image Shape: {img.shape} (Channels, Height, Width)")
    print(f" Label ID: {label}")

except Exception as e:
    print(f" Error: {e}")

print("[3/3] Data setup complete.")
