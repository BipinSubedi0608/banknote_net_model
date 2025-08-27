# Currency Recognition App

This project is a banknote currency recognition system using deep learning. It allows users to upload images of banknotes and predicts the currency type using a trained neural network.

## Features

- Upload banknote images via a web interface
- Predict currency type and confidence score
- Uses PyTorch and Streamlit for model inference and UI

## Requirements

- Python 3.8+
- [Virtual environment (venv)](https://docs.python.org/3/library/venv.html)
- See [`requirements.txt`](requirements.txt) for Python packages

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Create and activate a virtual environment:

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. Place your trained model weights in the appropriate folder (see [`src/app.py`](src/app.py) for path).
2. Run the Streamlit app:
   ```sh
   streamlit run src/app.py
   ```
3. Open the provided local URL in your browser.
4. Upload a banknote image and view the prediction.

## Project Structure

- `src/` - Source code
- `data/` - Raw and processed data
- `saved/` - Saved models and processed arrays

## How it works

- The model is trained to recognize different currencies from banknote images.
- The web app preprocesses the uploaded image and runs inference using the trained model.
- The predicted currency and confidence score are displayed to the user.

## License

MIT License

## Contact

For questions or contributions, open an issue or pull request on GitHub.
