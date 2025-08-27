RAW_FEATHER_DATA_PATH = "../data/banknote_net.feather"
RAW_CSV_DATA_PATH = "../data/banknote_net.csv"

PROCESSED_DATA_DIR = "../saved/processed/"
SAVED_MODELS_DIR = "../saved/models/"
EXAMPLE_IMAGES_DIR = "../data/examples/"

X_TRAIN_FILE = "X_train.npy"
Y_TRAIN_FILE = "y_train.npy"
X_TEST_FILE = "X_test.npy"
Y_TEST_FILE = "y_test.npy"

CLASSIFIER_MODEL_NAME = 'currency_classifier.pth'
ENCODER_MODEL_NAME = "banknote_net_encoder.onnx"

EXAMPLE_IMAGES_NAMES = [f"example_{i}.jpg" for i in range(1,5)]

CURRENCY_LABEL_MAP = {
    'AUD': 0,
    'BRL': 1,
    'CAD': 2,
    'EUR': 3,
    'GBP': 4,
    'IDR': 5,
    'INR': 6,
    'JPY': 7,
    'MXN': 8,
    'MYR': 9,
    'NNR': 10,
    'NZD': 11,
    'PHP': 12,
    'PKR': 13,
    'SGD': 14,
    'TRY': 15,
    'USD': 16
}