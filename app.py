import os

from flask import Flask, render_template, request
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn


app = Flask(__name__) #создаём веб-сервер

#устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#модель
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)#меняем последний слой:было: 1000 классов/стало: 2 (кот / собака)
model.load_state_dict(torch.load("model_best.pth", map_location=device))#загружаем обученные веса(ВАЖНО)
model.to(device) #переносим модель на gpu/cpu
model.eval() #отключает обучение, предсказания стабильнее

#трансформации
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

classes = ["cats","dogs"] # 0 / 1

#==============================================================================================================#

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filename = file.filename.lower()

            if not filename.endswith((".png", ".jpg", ".jpeg")):
                return "Ошибка: загрузи изображение"

            # путь сохранения
            filepath = os.path.join("static", filename)
            file.save(filepath)

            image_path = filepath

            # обработка
            image = Image.open(filepath).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                label = classes[predicted.item()]
                confidence = probs[0][predicted.item()].item()

                result = f"{label} ({confidence*100:.2f}%)"

    return render_template("index.html", result=result, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)