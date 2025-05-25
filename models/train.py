import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from models.cnn import get_pytorch_model, get_tensorflow_model
from utils.prep import get_pytorch_data, get_tensorflow_data

class Trainer:
    def __init__(self, model_type, train_data, test_data, lr, wd, epochs, device):
        self.epochs = epochs
        self.model_type = model_type
        self.device = device
        self.train_data = train_data
        self.test_data = test_data

        if model_type == 'pytorch':
            self.model = get_pytorch_model().to(device)
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=wd)
            self.criterion = nn.CrossEntropyLoss()
        else:  # tensorflow
            self.model = get_tensorflow_model()
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, weight_decay=wd),
                             loss='sparse_categorical_crossentropy',
                             metrics=['accuracy'])

    def train(self, save=True, plot=True):
        if self.model_type == 'pytorch':
            self.model.train()
            self.train_acc = []
            self.train_loss = []
            for epoch in range(self.epochs):
                total_loss = 0
                total_correct = 0
                total_samples = 0
                progress_bar = tqdm(self.train_data, desc=f"Epoch {epoch + 1}/{self.epochs}")
                for batch in progress_bar:
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    _, preds = outputs.max(1)
                    total_correct += (preds == labels).sum().item()
                    total_samples += labels.size(0)
                    total_loss += loss.item()
                    batch_accuracy = 100.0 * (preds == labels).sum().item() / labels.size(0)
                    progress_bar.set_postfix({'Batch Acc': f'{batch_accuracy:.2f}%'})
                avg_accuracy = 100.0 * total_correct / total_samples
                avg_loss = total_loss / len(self.train_data)
                self.train_acc.append(avg_accuracy)
                self.train_loss.append(avg_loss)
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%")
            if save:
                torch.save(self.model.state_dict(), "Jerome_teyi_model.torch")
            if plot:
                self.plot_training_history()
        else:  # tensorflow
            history = self.model.fit(self.train_data, epochs=self.epochs, validation_data=self.test_data, verbose=1)
            if save:
                self.model.save("Jerome_teyi_model.h5")
            if plot:
                plt.plot(history.history['loss'], label='Loss')
                plt.plot(history.history['accuracy'], label='Accuracy')
                plt.xlabel('Epoch')
                plt.title('Training Loss and Accuracy')
                plt.legend()
                plt.show()

    @torch.no_grad()
    def evaluate(self):
        if self.model_type == 'pytorch':
            self.model.eval()
            total_correct = 0
            total_samples = 0
            total_loss = 0
            for inputs, labels in tqdm(self.test_data, desc="Evaluating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, preds = outputs.max(1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                total_loss += loss.item() * labels.size(0)
            avg_loss = total_loss / total_samples
            accuracy = 100.0 * total_correct / total_samples
            print(f"Test Accuracy: {accuracy:.2f}% | Test Loss: {avg_loss:.4f}")
        else:
            results = self.model.evaluate(self.test_data, verbose=1)
            print(f"Test Accuracy: {results[1]*100:.2f}% | Test Loss: {results[0]:.4f}")

    def plot_training_history(self):
        if self.model_type == 'pytorch':
            epochs = range(1, len(self.train_loss) + 1)
            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss', color='tab:blue')
            ax1.plot(epochs, self.train_loss, color='tab:blue', label='Loss')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax2 = ax1.twinx()
            ax2.set_ylabel('Accuracy (%)', color='tab:red')
            ax2.plot(epochs, self.train_acc, color='tab:red', label='Accuracy')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            plt.title('Training Loss and Accuracy')
            plt.show()