# Currency Recognition App

This project is a banknote currency recognition system using deep learning. It allows users to upload images of banknotes and predicts the currency type using a trained neural network. This model is trained on the [Banknote_net](https://github.com/microsoft/banknote-net) dataset provided by **Microsoft**.

## Features

- Upload banknote images via a web interface
- Predict currency type and confidence score
- Uses PyTorch and Streamlit for model inference and UI

## Requirements

- Python 3.11.9
- [Virtual environment (venv)](https://docs.python.org/3/library/venv.html)
- See [`requirements.txt`](requirements.txt) for Python packages

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/BipinSubedi0608/banknote_net_model.git
   cd banknote_net_model
   ```

2. Create and activate a virtual environment:

   ```sh
   python -m venv <VENV_NAME>
   source <VENV_NAME>/bin/activate  # On Windows: <VENV_NAME>\Scripts\activate
   ```

   **NOTE:** Replace the `<VENV_NAME>` with the name you want for your virtual environment.<br>
   _THE NAME MUST CONTAIN `venv` SOMEWHERE IN IT._

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app from `src` directory:
   ```sh
   cd src
   streamlit run app.py
   ```
2. Open the provided local URL in your browser.
3. Upload a banknote image and view the prediction.

## Project Structure

- `src/` - Source code
- `data/` - Raw and processed data
- `saved/` - Saved models

## How it works

- The model is trained to recognize different currencies from banknote images using [Banknote_net](https://github.com/microsoft/banknote-net) dataset provided by **Microsoft**.
- The web app preprocesses the uploaded image and runs inference using the trained model.
- The predicted currency and confidence score are displayed to the user.

## License

Not licensed yet.

## Contact

For questions or contributions, open an issue on GitHub or visit the developer's [portfolio](https://bipinsubedi1.com.np) to contact.
