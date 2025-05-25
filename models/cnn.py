import torch
import torch.nn as nn
from torchvision import models as tv_models  # ✅ Alias pour torchvision

import tensorflow as tf
from tensorflow.keras import layers, models as tf_models  # ✅ Alias pour Keras


def get_pytorch_model():
    # Charger AlexNet avec les bons poids (évite le warning)
    model = tv_models.alexnet(weights=tv_models.AlexNet_Weights.DEFAULT)

    # Remplacer la première couche : 1 canal (entrée), 70 canaux (sortie)
    model.features[0] = nn.Conv2d(1, 70, kernel_size=5, stride=2, padding=2)

    # Adapter la deuxième couche qui s'attendait à 64 canaux
    model.features[3] = nn.Conv2d(70, 192, kernel_size=5, padding=2)

    # Geler les premières couches (pas toutes ! pour fine-tuning partiel)
    for param in model.features[:6].parameters():
        param.requires_grad = False

    # Dropout pour mieux généraliser
    model.classifier[5] = nn.Dropout(0.5)

    # Adapter la dernière couche fully connected pour 4 classes
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 4)

    return model



def get_tensorflow_model():
    model = tf_models.Sequential([
        # Conv1
        layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 1)),
        layers.MaxPooling2D((2, 2)),

        # Conv2
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Conv3
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Flatten + FC
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(4, activation='softmax')  # 4 classes
    ])
    return model
