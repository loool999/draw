from flask import Flask, request, jsonify, render_template
import base64
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

# Load or train the model
def load_model():
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/guess', methods=['POST'])
def guess():
    data = request.json
    image_data = data['image'].split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('L').resize((28, 28))
    image = np.array(image).reshape((1, 28, 28, 1)) / 255.0
    prediction = model.predict(image)
    guesses = np.argsort(prediction[0])[-3:][::-1]  # Top 3 guesses
    return jsonify({'guesses': [int(g) for g in guesses]})

if __name__ == '__main__':
    app.run(debug=True)
