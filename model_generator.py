# model_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import trimesh
import open3d as o3d
from torchvision import transforms
import clip
from transformers import AutoTokenizer, AutoModel

class ImageEncoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(512, latent_dim)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class TextEncoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.projection = nn.Linear(384, latent_dim)
        
    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return self.projection(embeddings)

class MeshDecoder(nn.Module):
    def __init__(self, latent_dim=512, max_vertices=1000, max_faces=1000):
        super().__init__()
        self.max_vertices = max_vertices
        self.max_faces = max_faces
        
        # Декодер для вершин
        self.vertex_decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, max_vertices * 3)  # x, y, z для каждой вершины
        )
        
        # Декодер для граней (индексы вершин)
        self.face_decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, max_faces * 3)  # 3 индекса на грань
        )
        
    def forward(self, latent):
        vertices = self.vertex_decoder(latent)
        faces = self.face_decoder(latent)
        
        # Решейп и нормализация
        vertices = vertices.view(-1, self.max_vertices, 3)
        vertices = torch.tanh(vertices)  # Нормализация к [-1, 1]
        
        faces = faces.view(-1, self.max_faces, 3)
        faces = torch.sigmoid(faces) * (self.max_vertices - 1)  # Индексы в диапазоне [0, max_vertices-1]
        faces = torch.round(faces).long()
        
        return vertices, faces

class MeshGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim = 512
        self.image_encoder = ImageEncoder(self.latent_dim)
        self.text_encoder = TextEncoder(self.latent_dim)
        self.mesh_decoder = MeshDecoder(self.latent_dim * 2)  # Объединяем image и text features
        
    def forward(self, image, text):
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text)
        
        # Объединяем features
        combined_features = torch.cat([image_features, text_features], dim=1)
        
        vertices, faces = self.mesh_decoder(combined_features)
        return vertices, faces

class MeshPostProcessor:
    def __init__(self, max_faces=1000):
        self.max_faces = max_faces
        
    def simplify_mesh(self, vertices, faces):
        """Упрощение меша до 1000 полигонов"""
        try:
 # Конвертируем в Open3D меш
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy())
            mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
            
            # Упрощаем меш
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=self.max_faces)
            
            # Возвращаем упрощенные вершины и грани
            return torch.tensor(np.asarray(mesh.vertices)), torch.tensor(np.asarray(mesh.triangles))
        except:
            # Fallback: берем первые max_faces граней
            return vertices, faces[:self.max_faces]
    
    def normalize_mesh(self, vertices):
        """Нормализация меша к единичному кубу"""
        vertices = vertices - vertices.mean(dim=0)
        max_val = torch.abs(vertices).max()
        if max_val > 0:
            vertices = vertices / max_val
        return vertices

class MeshGeneratorPipeline:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = MeshGenerator().to(device)
        self.post_processor = MeshPostProcessor()
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_weights(self, model_path):
        """Загрузка весов модели"""
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Модель загружена из {model_path}")
        except:
            print("Используется модель с случайными весами")
    
    def generate_from_image_and_text(self, image_path, text_description):
        """Генерация 3D модели по изображению и текстовому описанию"""
        # Загрузка и обработка изображения
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
        
        # Генерация
        with torch.no_grad():
            vertices, faces = self.model(image_tensor, [text_description])
        
        # Постобработка
        vertices, faces = self.post_processor.simplify_mesh(vertices[0], faces[0])
        vertices = self.post_processor.normalize_mesh(vertices)
        
        return vertices, faces
    
    def save_mesh(self, vertices, faces, output_path):
        """Сохранение меша в формате .obj"""
        mesh = trimesh.Trimesh(vertices=vertices.numpy(), faces=faces.numpy())
        mesh.export(output_path)
        print(f"Меш сохранен в {output_path}")
        
    def generate_and_save(self, image_path, text_description, output_path):
        """Полный пайплайн генерации и сохранения"""
        vertices, faces = self.generate_from_image_and_text(image_path, text_description)
        self.save_mesh(vertices, faces, output_path)
        return vertices, faces
