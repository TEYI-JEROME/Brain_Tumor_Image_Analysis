from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
import io
import tensorflow as tf
import numpy as np
from models.cnn import get_pytorch_model
import os

app = Flask(__name__)

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']


def load_pytorch_model():
    model_path = "Jerome_teyi_model.torch"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PyTorch model file not found at {model_path}")
    model = get_pytorch_model()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_tensorflow_model():
    model_path = "Jerome_teyi_model.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TensorFlow model file not found at {model_path}")
    return tf.keras.models.load_model(model_path)

pytorch_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image fournie'}), 400
    
    file = request.files['image']
    model_type = request.form.get('model_type', 'pytorch')
    
    try:
        img = Image.open(io.BytesIO(file.read())).convert('L')
        
        if model_type == 'pytorch':
            model = load_pytorch_model()
            img_tensor = pytorch_transform(img).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)[0] * 100
                pred = output.max(1)[1].item()
                prediction = CLASS_NAMES[pred]
                prob_dict = {CLASS_NAMES[i]: f"{probs[i].item():.2f}%" for i in range(len(CLASS_NAMES))}
        else:
            model = load_tensorflow_model()
            img_array = tf.keras.preprocessing.image.img_to_array(img.resize((224, 224)))
            img_array = img_array / 255.0
            img_array = img_array.reshape(1, 224, 224, 1)
            output = model.predict(img_array, verbose=0)
            pred = output.argmax()
            prediction = CLASS_NAMES[pred]
            probs = output[0] * 100
            prob_dict = {CLASS_NAMES[i]: f"{probs[i]:.2f}%" for i in range(len(CLASS_NAMES))}
        
        return jsonify({
            'prediction': prediction,
            'probabilities': prob_dict
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)