# train.py
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from model_generator import MeshGenerator

class MeshDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.samples = self.load_samples()
        
    def load_samples(self):
        """Загрузка датасета - здесь нужна ваша реализация"""
        # Пример структуры датасета:
         [{"image": r"C:\Users\user\Documents\3d neural network\datasets\fad.png", "text": "sword", "mesh": r"C:\Users\user\Documents\3d neural network\datasets\sw.obj" }]
        return []
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Загрузка изображения, текста и меша
        # Возврат в виде тензоров
        return sample

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Модель и оптимизатор
    model = MeshGenerator().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Датасет и загрузчик
    dataset = MeshDataset('path/to/dataset')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    # Цикл обучения
    for epoch in range(100):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            images = batch['image'].to(device)
            texts = batch['text']
            target_vertices = batch['vertices'].to(device)
            target_faces = batch['faces'].to(device)
            
            # Forward pass
            pred_vertices, pred_faces = model(images, texts)
            
            # Вычисление loss (упрощенное)
            vertex_loss = F.mse_loss(pred_vertices, target_vertices)
            face_loss = F.cross_entropy(pred_faces.view(-1, 1000), target_faces.view(-1))
            
            loss = vertex_loss + face_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}')

if __name__ == "__main__":
    train_model()
