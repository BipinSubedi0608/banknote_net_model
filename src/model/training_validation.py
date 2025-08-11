# import numpy as np
import torch
# import torch.optim as optim
# from torch import FloatTensor, LongTensor
# from torch.utils.data import TensorDataset, DataLoader
# from torch.nn import CrossEntropyLoss
# import matplotlib.pyplot as plt
# from model_definition import CurrencyClassifier
# from src.utils.constants import PROCESSED_DATA_DIR, SAVED_MODELS_DIR
from src.utils.helpers import set_cuda_if_available

set_cuda_if_available()
print(torch.cuda.current_device())