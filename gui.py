# gui.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

from model_generator import MeshGeneratorPipeline

class MeshVisualizer:
    def __init__(self):
        self.fig = None
        self.ax = None
        
    def visualize_mesh(self, vertices, faces, title="3D Model"):
        """Визуализация 3D меша с помощью matplotlib"""
        if self.fig is None:
            self.fig = plt.figure(figsize=(8, 6))
            self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.clear()
        
        # Отображаем меш
        self.ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                            triangles=faces, alpha=0.8, edgecolor='black')
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(title)
        
        # Устанавливаем равные масштабы осей
        max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(), 
                             vertices[:, 1].max()-vertices[:, 1].min(), 
                             vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0
        
        mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5
        
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.show()

class MeshGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Mesh Generator - AI")
        self.root.geometry("1200x800")
        
        # Инициализация пайплайна генерации
        self.generator = MeshGeneratorPipeline()
        
        # Инициализация визуализатора
        self.visualizer = MeshVisualizer()
        
        # Переменные
        self.image_path = tk.StringVar()
        self.text_description = tk.StringVar(value="3D модель объекта")
        self.output_path = tk.StringVar(value="output_mesh.obj")
        
        self.setup_ui()
        
    def setup_ui(self):
        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Заголовок
        title_label = ttk.Label(main_frame, text="🎮 Генератор 3D Моделей", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Левая панель - управление
        control_frame = ttk.LabelFrame(main_frame, text="Управление", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Загрузка изображения
        ttk.Label(control_frame, text="Изображение:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(control_frame, textvariable=self.image_path, width=40).grid(row=1, column=0, pady=5)
        ttk.Button(control_frame, text="Обзор...", command=self.browse_image).grid(row=1, column=1, pady=5)
        
        # Превью изображения
        self.image_preview = ttk.Label(control_frame, text="Превью изображения")
        self.image_preview.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Текстовое описание
        ttk.Label(control_frame, text="Текстовое описание:").grid(row=3, column=0, sticky=tk.W, pady=5)
        description_entry = ttk.Entry(control_frame, textvariable=self.text_description, width=40)
        description_entry.grid(row=4, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Примеры описаний
        examples_frame = ttk.Frame(control_frame)
        examples_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        ttk.Label(examples_frame, text="Примеры:").grid(row=0, column=0, sticky=tk.W)
        
        examples = [
            "Стул с четырьмя ножками и спинкой",
            "Автомобиль с кузовом и колесами", 
            "Чайник с ручкой и носиком",
            "Простая ваза для цветов"
        ]
        
        for i, example in enumerate(examples):
            btn = ttk.Button(examples_frame, text=example, 
                           command=lambda e=example: self.text_description.set(e))
            btn.grid(row=i+1, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Выходной файл
        ttk.Label(control_frame, text="Выходной файл:").grid(row=6, column=0, sticky=tk.W, pady=5)
        ttk.Entry(control_frame, textvariable=self.output_path, width=40).grid(row=7, column=0, pady=5)
        ttk.Button(control_frame, text="Обзор...", command=self.browse_output).grid(row=7, column=1, pady=5)
        
        # Кнопки генерации
        generate_frame = ttk.Frame(control_frame)
        generate_frame.grid(row=8, column=0, columnspan=2, pady=20)
        ttk.Button(generate_frame, text="🔄 Сгенерировать 3D Модель", 
                  command=self.generate_mesh, style="Accent.TButton").grid(row=0, column=0, pady=5)
        
        ttk.Button(generate_frame, text="👁️ Показать 3D", 
                  command=self.visualize_mesh).grid(row=1, column=0, pady=5)
        
        ttk.Button(generate_frame, text="💾 Сохранить OBJ", 
                  command=self.save_mesh).grid(row=2, column=0, pady=5)
        
        # Правая панель - информация и статус
        info_frame = ttk.LabelFrame(main_frame, text="Информация", padding="10")
        info_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Статус
        self.status_text = tk.Text(info_frame, height=15, width=50, wrap=tk.WORD)
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.status_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        # Статистика
        stats_frame = ttk.Frame(info_frame)
        stats_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        self.stats_label = ttk.Label(stats_frame, text="Статистика: Ожидание генерации...")
        self.stats_label.grid(row=0, column=0, sticky=tk.W)
        
        # Хранилище текущего меша
        self.current_vertices = None
        self.current_faces = None
        
        # Настройка весов колонок и строк
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        control_frame.columnconfigure(0, weight=1)
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
        
        self.log_status("🚀 Генератор 3D моделей готов к работе!")
        self.log_status("📝 Загрузите изображение и введите описание")
        self.log_status("⚙️ Ограничение: до 1000 треугольников")
        
    def browse_image(self):
        """Выбор файла изображения"""
        filename = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if filename:
            self.image_path.set(filename)
            self.show_image_preview(filename)
            
    def browse_output(self):
        """Выбор файла для сохранения"""
        filename = filedialog.asksaveasfilename(
            title="Сохранить 3D модель",
            defaultextension=".obj",
            filetypes=[("OBJ files", "*.obj"), ("All files", "*.*")]
        )
        if filename:
            self.output_path.set(filename)
            
    def show_image_preview(self, image_path):
        """Показ превью изображения"""
        try:
            image = Image.open(image_path)
            image.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(image)
            self.image_preview.configure(image=photo)
            self.image_preview.image = photo
        except Exception as e:
            self.log_status(f"❌ Ошибка загрузки изображения: {e}")
            
    def generate_mesh(self):
        """Генерация 3D модели в отдельном потоке"""
        if not self.image_path.get():
            messagebox.showerror("Ошибка", "Пожалуйста, выберите изображение")
            return
            
        if not self.text_description.get():
            messagebox.showerror("Ошибка", "Пожалуйста, введите описание")
            return
            
        # Запуск в отдельном потоке
        thread = threading.Thread(target=self._generate_mesh_thread)
        thread.daemon = True
        thread.start()
        
    def _generate_mesh_thread(self):
        """Поток для генерации меша"""
        try:
            self.log_status("🔄 Начало генерации 3D модели...")
            self.update_stats("Генерация...")
            
            # Генерация меша
            vertices, faces = self.generator.generate_from_image_and_text(
                self.image_path.get(),
                self.text_description.get()
            )
            
            self.current_vertices = vertices
            self.current_faces = faces
            
            # Обновление статистики
            num_vertices = len(vertices)
            num_faces = len(faces)
            
            stats_text = f"✅ Генерация завершена!\n"
            stats_text += f"📊 Вершин: {num_vertices}\n"
            stats_text += f"📊 Граней: {num_faces}\n"
            stats_text += f"📐 Размер: {num_faces} треугольников"
            
            self.update_stats(stats_text)
            self.log_status("✅ 3D модель успешно сгенерирована!")
            self.log_status(f"📊 Статистика: {num_vertices} вершин, {num_faces} граней")
            
        except Exception as e:
            self.log_status(f"❌ Ошибка генерации: {e}")
            self.update_stats("Ошибка генерации")
            
    def visualize_mesh(self):
        """Визуализация сгенерированного меша"""
        if self.current_vertices is None or self.current_faces is None:
            messagebox.showerror("Ошибка", "Сначала сгенерируйте 3D модель")
            return
            
        try:
            self.visualizer.visualize_mesh(
                self.current_vertices.numpy(),
                self.current_faces.numpy(),
                "Сгенерированная 3D Модель"
            )
            self.log_status("👁️ 3D модель отображена в отдельном окне")
        except Exception as e:
            self.log_status(f"❌ Ошибка визуализации: {e}")
            
    def save_mesh(self):
        """Сохранение меша в файл"""
        if self.current_vertices is None or self.current_faces is None:
            messagebox.showerror("Ошибка", "Сначала сгенерируйте 3D модель")
            return
            
        try:
            output_path = self.output_path.get()
            self.generator.save_mesh(self.current_vertices, self.current_faces, output_path)
            self.log_status(f"💾 Модель сохранена в: {output_path}")
        except Exception as e:
            self.log_status(f"❌ Ошибка сохранения: {e}")
            
    def log_status(self, message):
        """Логирование статуса"""
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
        
    def update_stats(self, stats):
        """Обновление статистики"""
        self.stats_label.configure(text=stats)

def main():
    root = tk.Tk()
    
    # Стилизация
    style = ttk.Style()
    style.configure("Accent.TButton", foreground="white", background="#0078D7")
    
    app = MeshGeneratorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
