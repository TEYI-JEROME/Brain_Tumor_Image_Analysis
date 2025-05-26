import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tensorflow as tf


def get_pytorch_data(batch_size=70):
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.ImageFolder('data/training/', transform=train_transform)
    test_data = datasets.ImageFolder('data/testing/', transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_tensorflow_data(batch_size=64):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=10,
        brightness_range=[0.8, 1.2],
        zoom_range=0.2
    )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'data/training/',
        target_size=(224, 224),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='sparse'
    )
    test_generator = test_datagen.flow_from_directory(
        'data/testing/',
        target_size=(224, 224),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='sparse'
    )

    return train_generator, test_generator