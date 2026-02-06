from flask import Flask, render_template, request
from model import init_brain
import os
import random
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

# --- 1. INITIALIZE FLASK FIRST (The Fix) ---
app = Flask(__name__)

# --- 2. CONFIGURATION ---
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SAMPLE_FOLDER'] = 'static/samples'
DEVICE = torch.device("cpu") 

# --- 3. CLASS NAMES ---
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

# --- 4. LOAD BRAINS SAFELY ---

# --- 4. LOAD BRAIN SAFELY ---
def load_efficientnet():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 37)
    try:
        model.load_state_dict(torch.load('pet_classifier_pro.pth', map_location=DEVICE))
    except:
        print("⚠️ EfficientNet weights not found.")
    model.eval()
    return model

# Initialize Global Model
brain1 = load_efficientnet()
ai_brain = init_brain() # Existing AI brain

# --- 5. TTA LOGIC ---
def process_image_tta(image_path):
# --- 5. IMAGE PROCESSING (Lite Version: No TTA) ---
def process_image(image_path):
    # Standard transform only - Reduces RAM usage by 4x
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0) # Add batch dimension manually


# --- 6. ROUTES ---
@app.route("/")
def home():
    my_bio = "I am a Computer Engineering student pivoting to AI. I build systems that think."
    my_projects = [
        {
            "name": "Smart Wristband for Fall Detection (FYP)",
            "tech": "ESP32C6, C++, Edge Impulse",
            "desc": "Award-winning IoT device that uses a self-trained AI model to detect falls in real-time.",
            "link": "https://github.com/Re-0z/slip"
        },
        {
            "name": "Neural Network from Scratch",
            "tech": "Python, NumPy",
            "desc": "A simple feedforward neural network built from the ground up without using any deep learning libraries, just pure math and code.",
            "link": "https://github.com/Re-0z/Basic-Neural-Network"
        },
        {
            "name": "Live AI Logic Gate",
            "tech": "Pure Python (No TensorFlow)",
            "desc": "Interactive demo. Type inputs and watch the neurons fire.",
            "link": "/ai-demo"
        },
        {
            "name": "The Pet Council AI",
            "tech": "PyTorch, Ensemble Learning",
            "desc": "Interactive classifier using 2 AI Brains + Test Time Augmentation.",
            "link": "/cat-demo",
            "github": "https://github.com/Re-0z/Pet-Council-AI"
        }
    ]
    
    return render_template("home.html", bio = my_bio, projects=my_projects)

@app.route("/ai-demo", methods=["GET", "POST"])
def ai_demo():
    prediction = None
    inputs = [0, 0, 0]

    if request.method == "POST":
        try:
            val1 = float(request.form.get("val1"))
            val2 = float(request.form.get("val2"))
            val3 = float(request.form.get("val3"))
            inputs = [val1, val2, val3]

            result = ai_brain.inspect(inputs)
            prediction = result

        except ValueError:
            prediction = {"error": "Invalid Input"}

    return render_template("ai_demo.html", prediction=prediction)

@app.route("/cat-demo", methods=["GET", "POST"])
def cat_demo():
    prediction = None
    image_url = None
    
    # Get Sample Images
    sample_images = []
    if os.path.exists(app.config['SAMPLE_FOLDER']):
        sample_images = os.listdir(app.config['SAMPLE_FOLDER'])

    if request.method == "POST":
        target_path = None
        
        # Handle File Upload or Sample Drag
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            target_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(target_path)
            image_url = target_path

        elif 'sample_name' in request.form:
            sample_name = request.form['sample_name']
            target_path = os.path.join(app.config['SAMPLE_FOLDER'], sample_name)
            image_url = target_path

        # Run AI
        if target_path:
            try:
                inputs = process_image(target_path)
                with torch.no_grad():
                    out1 = brain1(inputs)
                    final_prob = F.softmax(out1, dim=1).mean(dim=0)
                    
                    top_probs, top_classes = final_prob.topk(5)
                    results = []
                    for i in range(5):
                        results.append({
                            "breed": CLASS_NAMES[top_classes[i].item()],
                            "confidence": f"{top_probs[i].item() * 100:.1f}%"
                        })
                    
                    prediction = {
                        "top_result": results[0],
                        "all_results": results,
                        "uncertain": top_probs[0].item() < 0.4
                    }
            except Exception as e:
                prediction = {"error": str(e)}

    return render_template("cat_demo.html", prediction=prediction, image_url=image_url, sample_images=sample_images)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)