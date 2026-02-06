import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn
import os
import torch.nn.functional as F

# --- CONFIGS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH_1 = './models/pet_classifier_pro.pth'
MODEL_PATH_2 = './models/pet_classifier_resnet.pth'

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

# --- LOAD BRAINS ---
def load_efficientnet():
    print("Loading Brain 1 (EfficientNet)...")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 37)
    model.load_state_dict(torch.load(MODEL_PATH_1, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    return model

def load_resnet():
    print("Loading Brain 2 (ResNet50)...")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 37)
    checkpoint = torch.load(MODEL_PATH_2, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(DEVICE)
    model.eval()
    return model

# --- TTA GENERATOR (Create 4 ver of the image) ---
def process_image_tta(image_path):
    #Standard IMG
    t_standard = transforms.Compose([
        transforms.Resize((256, 256)), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    #Flip IMG
    t_flip = transforms.Compose([
        transforms.Resize((256, 256)), transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    #Rotate IMG
    t_rotate = transforms.Compose([
        transforms.Resize((256, 256)), transforms.CenterCrop(224),
        transforms.RandomRotation(degrees=(15, 15)),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    #Zoom/Crop IMG
    t_crop = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error: {e}")
        return None

    return torch.stack([
        t_standard(image), t_flip(image), t_rotate(image), t_crop(image)
    ]).to(DEVICE)

# --- MAIN EXEC ---
if __name__ == '__main__':
    model1 = load_efficientnet()
    model2 = load_resnet()

    print("--------------------------------")
    print(" ULTIMATE VERSION (Ensemble + TTA) READY")
    print(" (Checking all 37 Breeds | Scanning image 8 times)")

    while True:
        image_path = input("\nDrag and drop an image file here (or type 'exit): ").strip('"')

        if image_path.lower() == 'exit':
            break

        if not os.path.exists(image_path):
            print("File not found.")
            continue

        try:
            inputs = process_image_tta(image_path)
            if inputs is None: continue

            with torch.no_grad():
                out1 = model1(inputs) #Brain 1 Votes
                prob1 = F.softmax(out1, dim=1).mean(dim=0)

                out2 = model2(inputs) #Brain 2 Votes
                prob2 = F.softmax(out2, dim=1).mean(dim=0)

                final_prob = (prob1 + prob2) / 2

                top_probs, top_classes = final_prob.topk(5)
                
                print(f"\nAI Analysis Results:")
                print(f"---------------------------------")

                if top_probs[0].item() < 0.4:
                    print(f"UNCERTAIN (Best guess is low confidence)")

                for i in range(5):
                    breed = CLASS_NAMES[top_classes[i].item()]
                    conf = top_probs[i].item()*100
                    prefix = "1." if i == 0 else f" {i+1}."
                    print(f" {prefix} {breed}: {conf:.2f}%")
        
        except Exception as e:
            print(f"Error: {e}")