import torch
import numpy as np
import torchvision

model_name = "resnet18"
model = getattr(torchvision.models, model_name)(pretrained=True)
model = model.eval()

from PIL import Image
image_path = 'cat.png'
img = Image.open(image_path).resize((224, 224))

# Preprocess the image and convert to tensor
from torchvision import transforms

my_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
img = my_preprocess(img)
img = np.expand_dims(img, 0)

with torch.no_grad():
    torch_img = torch.from_numpy(img)
    output = model(torch_img)

    top1_torch = np.argmax(output.numpy())

print(top1_torch)

# export onnx

torch_out = torch.onnx.export(model, torch_img, 'resnet18.onnx', verbose=True, export_params=True)