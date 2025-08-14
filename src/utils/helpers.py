import torch

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

if __name__ == '__main__':
    device = get_device()
    print('Device set successfully as', device)