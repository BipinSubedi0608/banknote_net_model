import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import numpy as np


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)  # type: ignore # Add batch dimension (1, 3, 224, 224)
    
    # Move channels dimension to the last (1, 224, 224, 3)
    tensor = np.transpose(tensor, (0, 2, 3, 1))  # (1, 224, 224, 3)
    return tensor


def embed_image(image_path, device):
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights).features.to(device)
    model.eval()
    img_tensor = preprocess_image(image_path)
    with torch.no_grad():
        embedding = model(img_tensor)
        embedding = torch.nn.functional.adaptive_avg_pool2d(embedding, (1, 1))
        embedding = embedding.view(embedding.size(0), -1)  # Flatten
    return embedding.cpu().numpy().squeeze()


def predict_currency(embedding, model, device, label_encoder=None):
    embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(embedding_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        confidence = probs[0][pred_idx].item()
    if label_encoder:
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
    else:
        pred_label = pred_idx
    return pred_label, confidence