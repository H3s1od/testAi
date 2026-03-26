import os
from PIL import Image

folders = ["data/train/cats", "data/train/dogs", "data/val/cats", "data/val/dogs"]

for folder in folders:
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        try:
            img = Image.open(path)
            img.verify()  # проверка, что файл открывается
        except Exception:
            print(f"Удаляю повреждённый файл: {path}")
            os.remove(path)