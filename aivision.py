import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch import nn

# загрузка модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model_best.pth", map_location=device))
model.eval()

# трансформации
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# картинка
img_path = "test.jpg"
img = Image.open(img_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0)

# хук для получения feature map
features = []
def hook_fn(module, input, output):
    features.append(output)

# берем последний conv слой
model.layer4.register_forward_hook(hook_fn)

# forward
output = model(input_tensor)
pred_class = output.argmax().item()

# backward
model.zero_grad()
output[0, pred_class].backward()

# получаем градиенты
gradients = model.layer4[-1].conv2.weight.grad

# упрощённый heatmap
heatmap = features[0].detach().numpy()[0].mean(axis=0)

heatmap = np.maximum(heatmap, 0)
heatmap /= heatmap.max()

# накладываем на изображение
img_cv = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed = heatmap * 0.4 + img_cv

# показываем
plt.imshow(cv2.cvtColor(superimposed.astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()