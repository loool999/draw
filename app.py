from flask import Flask, request, jsonify, render_template, send_file
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
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

# logging
logging.basicConfig(level=logging.CRITICAL)

# Defineing folder
TRAINING_DATA_FOLDER = 'training_data'
PLOTS_FOLDER = 'plots'
BAD_WORDS_FILE = 'bad words.txt'

# Ensure the folders exist
os.makedirs(TRAINING_DATA_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# Load bad words from the text file
with open(BAD_WORDS_FILE, 'r') as f:
    bad_words = [line.strip() for line in f]

# simple neural network
class SimpleNN(nn.Module):
    def __init__(self, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(64 * 64, 128)
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
    global labels, model, optimizer
    
    images = []
    targets = []
    
    for label in labels:
        label_folder = os.path.join(TRAINING_DATA_FOLDER, label)
        for image_path in glob.glob(os.path.join(label_folder, '*.png')):
            image = Image.open(image_path).resize((64, 64)).convert('L')
            image_array = np.array(image).flatten() / 255.0
            images.append(image_array)
            targets.append(labels.index(label))
    
    if images:
        images = torch.tensor(np.array(images), dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.long)
        return images, targets
    return None, None

def retrain_model():
    images, targets = load_training_data()
    if images is None or targets is None:
        logging.warning("No training data available for retraining.")
        return
    
    dataset = torch.utils.data.TensorDataset(images, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    global model, optimizer
    model = SimpleNN(len(labels))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    training_losses = []
    for epoch in range(600):
        epoch_loss = 50.0
        for batch_images, batch_targets in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        training_losses.append(epoch_loss / len(dataloader))
        logging.info(f"Epoch {epoch+1}, Loss: {training_losses[-1]}")
    plt.figure()
    plt.plot(range(1, len(training_losses) + 1), training_losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(PLOTS_FOLDER, 'training_loss.png'))
    plt.close()

    logging.info("Model retraining complete")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_word', methods=['POST'])
def check_word():
    data = request.json
    word = data['word']
    if any(bad_word in word.lower() for bad_word in bad_words):
        return jsonify({'exists': True})
    else:
        return jsonify({'exists': False})

def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data['image']
    image_data = image_data.split(',')[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = image.resize((64, 64)).convert('L')

    image_array = np.array(image).flatten() / 255.0
    image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        confidences = torch.softmax(outputs, dim=1).numpy().flatten()

    predictions = sorted(zip(labels, confidences), key=lambda x: x[1], reverse=True)[:13]
    print(predictions[0])

    return jsonify({'guesses': [{'label': label, 'confidence': conf * 100} for label, conf in predictions]})

@app.route('/train', methods=['POST'])
def train():
    global model, criterion, optimizer, labels

    data = request.json
    image_data = data['image']
    label = data['label'].capitalize().strip() 
    # Word filter
    if not label.isalnum() and not label.isspace():
        return jsonify({'error': 'Invalid label. Please use only letters.'}), 401

    if any(word in label.lower() for word in bad_words):
        return jsonify({'error': 'Invalid label. Please use a different label.'}), 400

    logging.debug(f"Received label: {label}")
    print(f"Received label: {label}")
    
    image_data = image_data.split(',')[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = image.resize((64, 64)).convert('L')

    image_array = np.array(image).flatten() / 255.0
    image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)

    label_folder = os.path.join(TRAINING_DATA_FOLDER, label)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
        labels.append(label)
        labels.sort()
        model = SimpleNN(len(labels))
        criterion = nn.CrossEntropyLoss()  
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Saving the image
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

    retrain_model()

    return jsonify({'status': 'training complete'})

@app.route('/retrain', methods=['POST'])
def retrain():
    retrain_model()
    return jsonify({'status': 'retraining complete'})

@app.route('/plot')
def plot():
    plot_path = os.path.join(PLOTS_FOLDER, 'training_loss.png')
    if os.path.exists(plot_path):
        return send_file(plot_path, mimetype='image/png')
    else:
        return jsonify({'error': 'Plot not found'}), 404

if __name__ == '__main__':
    port = '8000'
    retrain_model()
    app.run(debug=True)
