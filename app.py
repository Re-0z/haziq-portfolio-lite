from flask import Flask, render_template, request
from model import init_brain
import os
# --- 1. INITIALIZE FLASK FIRST ---
app = Flask(__name__)

# --- 2. CONFIGURATION ---
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SAMPLE_FOLDER'] = 'static/samples'
# Initialize Global Model
ai_brain = init_brain() 

# --- 3. ROUTES ---
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
            "tech": "PyTorch, Ensemble Learning (Code Only)",
            "desc": "Interactive classifier code available on GitHub. Demo disabled for Lite version.",
            "link": "https://github.com/Re-0z/Pet-Council-AI"
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



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)