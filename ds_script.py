import os
import shutil
import random

# Пути (замени на свои!)
IMAGES_PATH = r"C:\Users\unbox\Desktop\Face_Detect_DataSet\images"  # Папка с изображениями
LABELS_PATH = r"C:\Users\unbox\Desktop\Face_Detect_DataSet\labels"  # Папка с аннотациями
OUTPUT_PATH = "cleaned_dataset"  # Куда сохранять train/val
TRAIN_RATIO = 0.8  # Доля train

# Функция проверки аннотации YOLO
def is_valid_annotation(label_path):
    with open(label_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    
    # Фильтруем только строки с 5 элементами (class, x, y, w, h)
    valid_lines = [line for line in lines if len(line.split()) == 5]

    if valid_lines:
        with open(label_path, "w") as f:
            f.writelines("\n".join(valid_lines) + "\n")  # Перезаписываем
        return True
    return False  # Некорректная аннотация

# Создаем выходные папки
for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT_PATH, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, split, "labels"), exist_ok=True)

# Собираем изображения
images = [f for f in os.listdir(IMAGES_PATH) if f.endswith((".jpg", ".png"))]
random.shuffle(images)  # Перемешиваем

if not images:
    print("в папке dataset/images нет изображений!")
    exit()

# Обрабатываем файлы
processed = 0  # Счетчик обработанных файлов

for img_file in images:
    img_path = os.path.join(IMAGES_PATH, img_file)
    label_path = os.path.join(LABELS_PATH, os.path.splitext(img_file)[0] + ".txt")

    # Если аннотации нет или она некорректна — удаляем изображение и аннотацию
    if not os.path.exists(label_path):
        print(f"Удаление {img_file}: нет аннотации {label_path}")
        os.remove(img_path)
        continue

    if not is_valid_annotation(label_path):
        print(f"Удаление {img_file}: аннотация {label_path} повреждена")
        os.remove(img_path)
        os.remove(label_path)
        continue

    # Разбиваем на train/val
    split = "train" if random.random() < TRAIN_RATIO else "val"

    # Пути назначения
    img_dest = os.path.join(OUTPUT_PATH, split, "images", img_file)
    label_dest = os.path.join(OUTPUT_PATH, split, "labels", os.path.splitext(img_file)[0] + ".txt")

    # Перемещаем файлы
    shutil.copy2(img_path, img_dest)
    shutil.copy2(label_path, label_dest)

    processed += 1

print(f"Обработано {processed} изображений.")