import torch
import torch.nn as nn
from torchvision import models as tv_models

import tensorflow as tf
from tensorflow.keras import layers, models as tf_models


def get_pytorch_model():
    
    model = tv_models.alexnet(weights=tv_models.AlexNet_Weights.DEFAULT)

    
    model.features[0] = nn.Conv2d(1, 70, kernel_size=5, stride=2, padding=2)

    
    model.features[3] = nn.Conv2d(70, 192, kernel_size=5, padding=2)

    
    for param in model.features[:6].parameters():
        param.requires_grad = False

    
    model.classifier[5] = nn.Dropout(0.5)

   
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 4)

    return model



def get_tensorflow_model():
    model = tf_models.Sequential([
    
        layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 1)),
        layers.MaxPooling2D((2, 2)),

        
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),

        
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),

        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(4, activation='softmax')
    ])
    return model
