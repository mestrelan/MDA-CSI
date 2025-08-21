import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import time
import pandas as pd

import os
import sys
import io
import subprocess as sp

import LoadData4


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, num_heads, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(input_size, dropout)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        # x tem a forma [batch_size, seq_len, num_channels]
        batch_size, seq_len, num_channels = x.size()

        # Aplicar codificação posicional nas amostras temporais
        x = self.pos_encoder(x)

        # Passar pelo Transformer Encoder
        x = self.transformer_encoder(x)

        # Agregar ao longo da dimensão do comprimento da sequência
        x = x.mean(dim=1)  # Média ao longo da sequência temporal

        # Passar pela camada fully connected
        x = self.fc(x)

        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs):
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)

            outputs = outputs.squeeze(1)
            
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        average_train_loss = running_loss / len(train_loader)
        train_losses.append(average_train_loss)
        
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {average_train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    # Plotting the validation accuracy and loss #descomentar tudo
    epochs = range(1, num_epochs + 1)
    
    #plt.figure(figsize=(12, 5))
    
    #plt.subplot(1, 2, 1)
    #plt.plot(epochs, train_losses, 'b', label='Training loss')
    #plt.plot(epochs, val_losses, 'r', label='Validation loss')
    #plt.title('Training and validation loss')
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.legend()
    
    #plt.subplot(1, 2, 2)
    #plt.plot(epochs, val_accuracies, 'b', label='Validation accuracy')
    #plt.title('Validation accuracy')
    #plt.xlabel('Epochs')
    #plt.ylabel('Accuracy')
    #plt.legend()
    
    #plt.tight_layout()
    #plt.savefig('training_validation_metrics.png')
    #plt.show()
    listas_treinamento = epochs, train_losses, val_losses, val_accuracies
    
    return listas_treinamento

def validate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            
            loss = criterion(outputs, labels.float())
            
            running_loss += loss.item()
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.float()).sum().item()

    average_loss = running_loss / len(data_loader)
    accuracy = correct / total

    return average_loss, accuracy

def test_model(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            
            loss = criterion(outputs, labels.float())
            
            running_loss += loss.item()
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    average_loss = running_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    #with open('test_metrics.txt', 'w') as f:
    #    f.write(f'Test Loss: {average_loss:.4f}\n')
    #    f.write(f'Accuracy: {accuracy:.4f}\n')
    #    f.write(f'Precision: {precision:.4f}\n')
    #    f.write(f'Recall: {recall:.4f}\n')
    #    f.write(f'F1 Score: {f1:.4f}\n')
    #    f.write(f'Confusion Matrix:\n{conf_matrix}\n')

    return average_loss, accuracy, precision, recall, f1, conf_matrix


if __name__=="__main__":
    # Definindo hiperparâmetros e dispositivos
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(input_size=48, output_size=1, num_heads=3, num_layers=2).to(device)#num_heads=8, num_layers=6
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Definindo data loaders para treinamento, validação e teste
    train_loader = torch.load("train_loader.pt")  # Defina seu DataLoader de treinamento
    val_loader = torch.load("val_loader.pt")    # Defina seu DataLoader de validação
    test_loader = torch.load("test_loader.pt")   # Defina seu DataLoader de teste
    
    
    listas_treinamento=train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 100)
         
    
    test_loss, test_accuracy, test_precision, test_recall, test_f1, test_conf_matrix = test_model(model, test_loader, criterion, device)

     
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test Precision: {test_precision:.4f}')
    print(f'Test Recall: {test_recall:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}')
    print(f'Test Confusion Matrix:\n{test_conf_matrix}')
    #return test_loss, test_accuracy, test_precision, test_recall, test_f1, test_conf_matrix, temp_train_eval,temp_test,listas_treinamento,train_gpu_info,test_gpu_info,pessoas_treino, pessoas_validacao, pessoas_teste
    
