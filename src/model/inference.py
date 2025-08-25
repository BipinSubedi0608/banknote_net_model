import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import onnxruntime

from model.model_definition import CurrencyClassifier
from utils.helpers import get_currency_from_label
from utils.constants import SAVED_MODELS_DIR, ENCODER_MODEL_NAME


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
    tensor = transform(image).unsqueeze(0) # type: ignore  # (1, 3, 224, 224)
    tensor = tensor.permute(0, 2, 3, 1)     # (1, 224, 224, 3)
    return tensor.numpy().astype(np.float32)


def get_embedding(img):
    encoder_model_path = SAVED_MODELS_DIR + ENCODER_MODEL_NAME
    session = onnxruntime.InferenceSession(encoder_model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    onnx_output = session.run([output_name], {input_name: img})
    embedding = onnx_output[0]
    embedding_tensor = torch.tensor(embedding).float()
    return embedding_tensor


def predict_currency(embedding: torch.Tensor, classifier_model: CurrencyClassifier, device: torch.device):
    classifier_model.eval()
    with torch.no_grad():
        output = classifier_model(embedding.to(device))
        probs = torch.softmax(output, dim=1)
        pred_label = torch.argmax(output, dim=1).item().__int__()
        confidence = probs[0][pred_label].item()
    print(pred_label)
    currency = get_currency_from_label(pred_label)
    
    return currency, confidence