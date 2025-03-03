from ultralytics import YOLO
import torch

if __name__ == "__main__":
    # Проверим, доступен ли CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используемое устройство: {device}")

    # Загружаем модель на GPU
    model = YOLO("yolo11n.pt").to(device)
    print("Модель загружена на:", next(model.parameters()).device)

    # Обучение модели
    result = model.train(data="dataset.yaml", epochs=200, batch=64, imgsz=640, device=device)

    # Инференс на изображении
    result = model(r"C:\Users\unbox\Desktop\test_002\images\test\5.jpg", device=device)
    
    # Отображение результатов
    result[0].show()
