import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Указываем пути к файлам YOLO
YOLO_CONFIG_PATH = 'model/yolov4.cfg'
YOLO_WEIGHTS_PATH = 'model/yolov4.weights'
YOLO_CLASSES_PATH = os.path.join("model", "coco.names")

# Загружаем классы из coco.names
with open(YOLO_CLASSES_PATH, 'r') as f:
    classes = f.read().strip().split('\n')

# Загружаем модель YOLO
net = cv2.dnn.readNet(YOLO_WEIGHTS_PATH, YOLO_CONFIG_PATH)

# Папка для сохранения загруженных изображений
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return "YOLO Face Detection API"

@app.route('/static/uploads', methods=['POST'])
def upload_file():
    # Проверяем, был ли загружен файл
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Сохраняем файл
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Обработка изображения
    image = cv2.imread(file_path)
    height, width = image.shape[:2]

    # Генерируем входной blob для YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Отправляем blob через сеть
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    # Список объектов, обнаруженных на изображении
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Убираем дубликаты
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Рисуем прямоугольники на изображении
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Сохраняем изображение с результатами
    output_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + file.filename)
    cv2.imwrite(output_file_path, image)

    # Отправляем результат пользователю
    return send_from_directory(app.config['UPLOAD_FOLDER'], 'result_' + file.filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)