import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import mobilenet_v2


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
    tensor = transform(image)
    return tensor.unsqueeze(0) # type: ignore


def embed_image(image_path, device):
    model = mobilenet_v2(pretrained=True).features.to(device)
    model.eval()
    img_tensor = preprocess_image(image_path).to(device)
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