import torch
import time
import os
import warnings
from tqdm import tqdm
from torchvision.models import resnet18
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from datetime import datetime

warnings.filterwarnings("ignore")

if __name__ == '__main__':  #  всё внутри этого блока

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Устройство: {device}")
    if device.type == "cuda":
        print(f" GPU: {torch.cuda.get_device_name(0)}")

    num_workers = os.cpu_count()
    print(f"CPU ядер: {num_workers}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_data = datasets.ImageFolder("data/train", transform=transform)
    val_data   = datasets.ImageFolder("data/val",   transform=transform)

    train_loader = DataLoader(
        train_data, batch_size=32, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_data, batch_size=32,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 10
    best_acc = 0
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        loop = tqdm(
            train_loader,
            desc=f" Epoch {epoch+1}/{EPOCHS}",
            leave=False,
            ncols=100
        )

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            train_acc = 100 * correct / total

            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{train_acc:.1f}%")

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        minutes = int(epoch_time // 60)
        seconds = int(epoch_time % 60)

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        print(f"{'─'*55}")
        print(f"  Epoch {epoch+1}/{EPOCHS} завершена за {minutes}м {seconds}с")
        print(f"  Train  | loss: {avg_loss:.4f} | acc: {train_acc:.1f}%")
        print(f"  Val    | loss: {avg_val_loss:.4f} | acc: {val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "model_best.pth")
            print(f"Новый рекорд! Сохранено: {val_acc:.1f}%")

        print(f"{'─'*55}\n")

    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    print(f"Обучение завершено за {minutes} мин {seconds} сек")
    print(f"Лучшая точность: {best_acc:.2f}%")