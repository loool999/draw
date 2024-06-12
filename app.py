from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import logging
import shutil

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Define the folder for saving training images
TRAINING_DATA_FOLDER = 'training_data'

# Ensure the training data folder exists
if not os.path.exists(TRAINING_DATA_FOLDER):
    os.makedirs(TRAINING_DATA_FOLDER)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, criterion, and optimizer
labels = sorted([d for d in os.listdir(TRAINING_DATA_FOLDER) if os.path.isdir(os.path.join(TRAINING_DATA_FOLDER, d))])
model = SimpleNN(len(labels))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def load_training_data():
    """Load training data from the training_data folder."""
    global labels, model, optimizer
    
    images = []
    targets = []
    
    for label in labels:
        label_folder = os.path.join(TRAINING_DATA_FOLDER, label)
        for image_path in glob.glob(os.path.join(label_folder, '*.png')):
            image = Image.open(image_path).resize((28, 28)).convert('L')
            image_array = np.array(image).flatten() / 255.0
            images.append(image_array)
            targets.append(labels.index(label))
    
    if images:
        images = torch.tensor(images, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.long)
        return images, targets
    return None, None

def retrain_model():
    """Retrain the model with the current training data."""
    images, targets = load_training_data()
    if images is None or targets is None:
        logging.warning("No training data available for retraining.")
        return
    
    dataset = torch.utils.data.TensorDataset(images, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = SimpleNN(len(labels))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(5):
        for batch_images, batch_targets in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
    
    logging.info("Model retraining complete")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data['image']
    image_data = image_data.split(',')[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = image.resize((28, 28)).convert('L')

    image_array = np.array(image).flatten() / 255.0
    image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        confidences = torch.softmax(outputs, dim=1).numpy().flatten()

    predictions = sorted(zip(labels, confidences), key=lambda x: x[1], reverse=True)

    return jsonify({'guesses': [{'label': label, 'confidence': conf * 100} for label, conf in predictions]})

@app.route('/train', methods=['POST'])
def train():
    global model, criterion, optimizer, labels

    data = request.json
    image_data = data['image']
    label = data['label'].capitalize()  # Capitalize the first letter

    logging.debug(f"Received label: {label}")

    image_data = image_data.split(',')[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = image.resize((28, 28)).convert('L')

    image_array = np.array(image).flatten() / 255.0
    image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)

    label_folder = os.path.join(TRAINING_DATA_FOLDER, label)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
        labels.append(label)
        labels.sort()
        model = SimpleNN(len(labels))  # Update model to reflect new number of classes
        criterion = nn.CrossEntropyLoss()  # Reinitialize criterion
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Reinitialize optimizer
    
    # Save the image with a unique name
    image_path = os.path.join(label_folder, f"{label}_{len(os.listdir(label_folder))}.png")
    image.save(image_path)
    
    label_index = labels.index(label)
    label_tensor = torch.tensor([label_index], dtype=torch.long)

    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(image_tensor)
        loss = criterion(outputs, label_tensor)
        loss.backward()
        optimizer.step()
    
    logging.debug("Training complete")

    return jsonify({'status': 'training complete'})

@app.route('/retrain', methods=['POST'])
def retrain():
    """Endpoint to manually trigger retraining."""
    retrain_model()
    return jsonify({'status': 'retraining complete'})

if __name__ == '__main__':
    # Load existing training data and retrain model at startup
    retrain_model()
    app.run(debug=True)
