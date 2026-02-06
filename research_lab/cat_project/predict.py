import torch
import torchvision.transforms as transforms
from torchvision import models, datasets
from PIL import Image
import torch.nn as nn
import os
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = './models/pet_classifier_pro.pth'

# Load class names
print("Loading breed names...")
try:
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
except Exception as e:
    print(f"Error loading class names: {e}")
    exit()

#Rebuilding the brain
def load_model():
    print("Rebuilding model architecture...")
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 37)

    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print(" Weights loaded successfully.")
    except Exception as e:
        print(f" Error loading weights: {e}")
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

if __name__ == "__main__":
    model = load_model()
    print(" Model loaded successfully!")
    print("----------------------------------")

    while True:
        image_path = input("\nDrag and drop an image file here (or type 'exit'): ").strip('"')
        if image_path.lower() == 'exit':
            break
        if not os.path.exists(image_path):
            print(" File not found. Please try again.")
            continue
        try:
            tensor = process_image(image_path)
            with torch.no_grad():
                outputs = model(tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_prob, top_class = probabilities.topk(3, dim=1)
                
                print(f"\nAI Analysis:")
                for i in range(3):
                    breed = CLASS_NAMES[top_class[0][i].item()]
                    conf = top_prob[0][i].item() * 100
                    print(f" {i+1}. {breed}: {conf:.2f}%")

            if conf < 50:
                print(" I am not very sure about this one...")
        
        except Exception as e:
            print(f" Error processing image: {e}")