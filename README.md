# AI Drawing Guesser

AI Drawing Guesser is a web application that allows users to draw on a canvas, and an AI model tries to guess what the drawing represents.

## Features

- Draw on a canvas and get real-time predictions from the AI model.
- Train the AI model with new drawings and labels.

## Requirements

- Python 3.7+
- Flask
- Flask-CORS
- NumPy
- OpenCV
- Pillow
- PyTorch
- scikit-learn
- Matplotlib

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/loool999/draw.git
    cd ai-drawing-guesser
    ```

2. Set up a virtual environment and activate it:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the dependencies:

    ```bash
    pip install flask flask-cors pillow numpy torch torchvision watchdog matplotlib
    ```

## Usage

1. Run the Flask application:

    ```bash
    python app.py
    ```

2. Open your web browser and go to `http://127.0.0.1:5000`.

3. Draw on the canvas and see the AI model's predictions.

4. To train the model with a new drawing and label, enter the label in the textbox and click "Send".

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
