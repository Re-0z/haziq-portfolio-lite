import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn
import os
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = './models/pet_classifier_pro.pth'

CLASS_NAMES = [
    'Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 
    'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair', 'Chihuahua', 
    'Egyptian Mau', 'English Cocker Spaniel', 'English Setter', 'German Shorthaired', 
    'Great Pyrenees', 'Havanese', 'Japanese Chin', 'Keeshond', 'Leonberger', 
    'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 'Persian', 'Pomeranian', 
    'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard', 'Samoyed', 'Scottish Terrier', 
    'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier', 'Wheaten Terrier', 
    'Yorkshire Terrier'
]

#------LOAD MODEL------
def load_model():
    print("Loading EfficientNet-B0...")
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 37)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        print("Weights loaded.")
    except Exception as e:
        print(f"Error: {e}")
        exit()
    
    model.to(DEVICE)
    model.eval()
    return model

#-----------TTA PRE-PROCESSING-----------
def process_image_tta(image_path):
    base_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    flip_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    rotate_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomRotation(degrees=(15, 15)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    raw_image = Image.open(image_path).convert('RGB')

    img1 = base_transform(raw_image)
    img2 = flip_transform(raw_image)
    img3 = rotate_transform(raw_image)

    return torch.stack([img1, img2, img3]).to(DEVICE)

if __name__ == "__main__":
    model = load_model()
    print("---------------------------------------")
    print("TTA ENGINE READY (Checking 3 variations per image)")

    while True:
        image_path = input("\nDrag and drop an image file here (or type 'exit')").strip('"')

        if image_path.lower() == 'exit':
            break

        if not os.path.exists(image_path):
            print(" File not found.")
            continue

        try:
            input_batch = process_image_tta(image_path)

            with torch.no_grad():
                outputs = model(input_batch)
                probs = F.softmax(outputs, dim=1)
                avg_probs = torch.mean(probs, dim=0)
                top_prob, top_class = avg_probs.topk(1)
                breed = CLASS_NAMES[top_class.item()]
                confidence = top_prob.item() * 100

                sphynx_idx = CLASS_NAMES.index('Sphynx')
                rb_idx = CLASS_NAMES.index('Russian Blue')

                print(f"\nAI Consensus:")
                print(f" Winner: {breed} ({confidence:.2f}%)")
                print(f" --------------------------------")
                print(f" VS Check:")
                print(f" Sphynx Score: {avg_probs[sphynx_idx].item()*100:.2f}%")
                print(f" Russian Blue Score: {avg_probs[rb_idx].item()*100:.2f}%")
        
        except Exception as e:
            print(f"Error: {e}")