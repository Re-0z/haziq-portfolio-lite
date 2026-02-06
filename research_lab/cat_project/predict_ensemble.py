import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn
import os
import torch.nn.functional as F

#Configs
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

def load_efficientnet():
    print("Loading Brain 1 (EfficientNet)...")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 37)
    model.load_state_dict(torch.load(MODEL_PATH_1, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model

def load_resnet():
    print("Loading Brain 2 (Resnet 50)...")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 37)
    try:
        model.load_state_dict(torch.load(MODEL_PATH_2, map_location=DEVICE, weights_only=True))
    except FileNotFoundError:
        print("ResNet model not found! You need to train it first.")
        exit()
    model.to(DEVICE)
    model.eval()
    return model

def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(DEVICE)

# --- MAIN EXEC ---
if __name__ == '__main__':
    model1 = load_efficientnet()
    model2 = load_resnet()

    print("--------------------------------")
    print(" THE COUNCIL IS ASSEMBLED")

    while True:
        image_path = input("\nDrag and drop an image file here (or type 'exit'): ").strip('"')

        if image_path.lower() == 'exit':
            break
        if not os.path.exists(image_path):
            print("File not found.")
            continue

        try:
            tensor = process_image(image_path)

            with torch.no_grad():
                out1 = model1(tensor)
                prob1 = F.softmax(out1, dim=1)

                out2 = model2(tensor)
                prob2 = F.softmax(out2, dim=1)

                final_prob = (prob1 + prob2) / 2
                top_prob, top_class = final_prob.topk(1)
                breed = CLASS_NAMES[top_class.item()]
                confidence = top_prob.item() * 100

                sphynx_idx = CLASS_NAMES.index('Sphynx')
                rb_idx = CLASS_NAMES.index('Russian Blue')

                print(f"\nCouncil Verdict: {breed.upper()}")
                print(f" Confidence: {confidence:.2f}%")
                print(f" -----------------------------")
                print(f" EfficientNet thought: {CLASS_NAMES[prob1.topk(1)[1].item()]}")
                print(f" ResNet50 thought: {CLASS_NAMES[prob2.topk(1)[1].item()]}")
                print(f" -----------------------------")
                print(f" Sphynx Score: {final_prob[0][sphynx_idx].item()*100:.2f}%")
                print(f" Russian Blue Score: {final_prob[0][rb_idx].item()*100:.2f}%")
        
        except Exception as e:
            print(f"Error: {e}")