# 🧠 DeepFake Detection using AI

A deep learning–based application to detect **deepfake videos** using **CNN + Graph Neural Network (GNN)** models.  
Built with **PyTorch**, **OpenCV**, and **Streamlit**.

---

## 🚀 Features
- Detects deepfake faces in uploaded videos.
- Uses CNN for feature extraction and GNN for relational reasoning.
- Real-time video frame analysis.
- User-friendly Streamlit interface.

---

## 🧩 Tech Stack
- Python  
- PyTorch  
- OpenCV  
- Streamlit  
- NumPy  
- Matplotlib  

---

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/sanjanak846/DeepFake-Detection.git
   cd DeepFake-Detection
2.Create a virtual environment (recommended):
python -m venv venv
venv\Scripts\activate      # for Windows

Install dependencies:
pip install -r requirements.txt

3.Run the App
streamlit run app.py

4.Steps to run in Streamlit
In your VS Code terminal (already open at the correct path), enter:
py -m streamlit run app.py


Then open the displayed local URL in your browser.

📊 Model Overview

The project uses:
i)CNN layers for spatial feature extraction.
ii)Graph Neural Networks (GNN) for temporal and relational understanding.
iii)Trained on open deepfake datasets for binary classification (real vs fake).

🧪 Example Output

<img width="1729" height="812" alt="image" src="https://github.com/user-attachments/assets/e469a6e7-d8de-48c1-8b31-bddcc7fc9325" />

💡 Future Improvements

Support for multiple faces per frame
Live webcam deepfake detection
Model optimization for faster inference

👩‍💻 Author

Chandana L D
