from flask import Flask, request, render_template
import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import numpy as np
from deepface import DeepFace

app = Flask(__name__)

# Load MobileNetV2 for Image Recognition
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Define Image Preprocessing
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Image Recognition Route
@app.route('/recognize', methods=['POST'])
def recognize_image():
    if 'file' not in request.files:
        return "No file uploaded."
    
    file = request.files['file']
    image = Image.open(file.stream)
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = output.max(1)
        labels = ["Label 1", "Label 2", "Label 3"]  # Add actual labels here
        return f"Prediction: {labels[predicted.item()]}"

# Deepfake Detection Route
@app.route('/detect_deepfake', methods=['POST'])
def detect_deepfake():
    if 'file' not in request.files:
        return "No file uploaded."
    
    file = request.files['file']
    file_path = f"./uploads/{file.filename}"
    file.save(file_path)

    result = DeepFace.analyze(file_path, actions=['emotion', 'age', 'gender'])
    return str(result)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
