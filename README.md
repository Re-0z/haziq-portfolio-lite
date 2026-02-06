# 🚀 Haziq's Portfolio (Lite Version)

**A super-lightweight, cloud-optimized portfolio dashboard.**
*Designed for instant loading on free-tier hosting (Render).*

---

## 📖 About This Project
This is the **"Lite"** hub for my work. Instead of hosting heavy AI models explicitly, this serves as a fast, responsive gateway to my other projects. It runs on a minimal **Flask** backend with **zero** heavy machine learning dependencies to ensure maximum speed and stability.

### ⚡ Key Features
*   **Pure Python Logic Gate:** A live, interactive neural network built from scratch (no libraries) to demonstrate the math behind AI.
*   **Project Showcase:** Links to my featured works like *The Pet Council* and *Smart Wristband*.
*   **Production Ready:** Configured with `gunicorn` for stable deployment.

---

## 🧠 Live Demos

### 1. Live AI Logic Gate
An interactive neural network that runs on **pure Python math**.
*   **Tech:** Python (No TensorFlow/PyTorch)
*   **Description:** Watch how a neural network "thinks" by typing inputs and seeing the neurons fire in real-time.

### 2. The Pet Council AI (Link Only)
My advanced breed classifier project is hosted separately to preserve resources.
*   **Link:** [View on GitHub](https://github.com/Re-0z/Pet-Council-AI)

---

## 🛠️ Tech Stack
*   ** Backend:** Flask, Gunicorn
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