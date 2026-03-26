import os

import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn
from PIL import Image
import sys

# Загрузка модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model_best.pth", map_location=device))
model = model.to(device)
model.eval()

# Те же трансформации что при обучении
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

classes = ["cats", "dogs"]  # должно совпадать с папками в data/train

def predict_folder(folder_path):
    images = [f for f in os.listdir(folder_path)
              if f.endswith(('.jpg', '.jpeg', '.png'))]

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]  # вероятности в %
        predicted = torch.argmax(probs).item()

    print(f"\n  Файл: {image_path}")
    print(f" Кошка: {probs[0]*100:.1f}%")
    print(f" Собака: {probs[1]*100:.1f}%")
    print(f" Ответ: {' Кошка' if predicted == 0 else ' Собака'}\n")

# Запуск: python test.py моя_картинка.jpg

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python test.py test/картинка.jpg   — одна картинка")
        print("  python test.py test/               — вся папка")
    else:
        path = sys.argv[1]
        if os.path.isdir(path):
            predict_folder(path)  # если папка — все картинки
        else:
            predict(path)         # если файл — одна картинка