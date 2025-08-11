import torch

def set_cuda_if_available():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device