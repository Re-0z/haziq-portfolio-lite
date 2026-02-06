# 🚀 Haziq's Portfolio (Lite Version)

**A lightweight, cloud-optimized version of my personal portfolio and AI showcase.**
*Optimized for free-tier deployment (e.g., Render) with reduced memory footprint.*

---

## 📖 About This Version
This repository contains a **"Lite"** version of my original portfolio. The key difference is the optimization of the AI backend to fit within strict memory limits (512MB RAM).

### ⚡ Key Optimizations
*   **Single-Brain Architecture:** The "Pet Council" module now runs exclusively on **EfficientNet-B0**. The ResNet50 model was removed to save memory while maintaining high accuracy for most cases.
*   **Production Ready:** Configured with `gunicorn` for stable cloud serving.
*   **No Docker Required:** Simplified for direct Python environment deployment.

---

## 🧠 Featured Projects

### 1. The Pet Council AI (Lite)
An interactive pet breed classifier that uses deep learning to identify 37 different breeds of cats and dogs.
*   **Model:** EfficientNet-B0 (PyTorch)
*   **Technique:** Test Time Augmentation (TTA) - The AI scans every image 4 times (Original, Flip, Rotate, Crop) to ensure robustness.
*   **Status:** Active & Deployable on Free Tier.

### 2. Smart Wristband for Fall Detection (Link)
A sneak peek at my IoT project using ESP32C6 and Edge Impulse.

### 3. Neural Network from Scratch (Link)
A demonstration of building AI using only NumPy and Math, without deep learning libraries.

---

## 🛠️ Tech Stack
*   ** Backend:** Flask, Gunicorn
*   ** AI/ML:** PyTorch (CPU only), Torchvision
*   ** Frontend:** HTML5, CSS3, JavaScript
*   ** Deployment:** Render (via `render.yaml`)

---

## 🚀 Quick Start (Local)

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Application**
    ```bash
    # For Windows
    python app.py
    
    # OR using Gunicorn (Linux/Mac/Git Bash)
    gunicorn app:app
    ```

3.  **Visit** `http://localhost:5000`