# E-commerce Review Sentiment Classifier

## Project Overview

This project implements a sentiment classification system for e-commerce product reviews using a fine-tuned BERT model. The system classifies reviews into multiple sentiment categories (e.g., Negative, Neutral, Positive, Very Positive) based on the text content of the review.

## Project Structure

```bash
E-Commerce_text_classification/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── bert_classifier.py
│   ├── models-weight/
│   │   └── sentiment_BERT_ecommerce-review_pytorch.pth
│   ├── static/
│   │   ├── script.js
│   │   └── style.css
│   ├── templates/
│   │   ├── index.html
│   │   └── predict.html
│   ├── utils/
│   │   ├── __init__.py
│   │   └── predict.py
│   ├── __init__.py
│   └── main.py
│
├── model/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
│
├── screenshots/
│   ├── app-command.png
│   ├── home.png
│   ├── eval.png
│   └── prediction.png
│
├── scripts/
│   ├── train_model.py
│   └── run_app.sh
│
├── requirements.txt
└── README.md
```

## Components

### 1. Model (model/)

- `config.py`: Contains configuration settings for the model, including hyperparameters and device settings.
- `dataset.py`: Defines the `EcommerceReviewDataset` class for handling the review data.
- `model.py`: Implements the `SentimentClassifierBert` class, which is a BERT-based model for sentiment classification.
- `train.py`: Contains the training loop and evaluation logic for the model.
- `utils.py`: Includes utility functions for data loading, preprocessing, and other helper functions.

### 2. Application (app/)

- `main.py`: The entry point for the FastAPI application.
- `api/routes.py`: Defines the API routes for the sentiment classification endpoint.
- `models/bert_classifier.py`: A copy of the model definition for use in the API.
- `utils/predict.py`: Contains the prediction logic used by the API.
- `static/`: Contains static files (CSS and JavaScript) for the web interface.
- `templates/`: Contains HTML templates for the web interface.

### 3. Scripts (scripts/)

- `train_model.py`: A script to train the sentiment classification model.
- `run_app.sh`: A shell script to start the FastAPI server.

## Model Architecture

The sentiment classifier is based on the BERT (Bidirectional Encoder Representations from Transformers) model. Specifically:

- We use the 'bert-base-cased' pretrained model as the base.
- The last 6 layers of BERT are fine-tuned for our specific task, while earlier layers remain frozen.
- A dropout layer (p=0.25) is applied after the BERT output for regularization.
- A final linear layer maps the BERT output to the number of sentiment classes.

## Data Preparation

The model expects input data in the form of a CSV file with two columns: 'category' (sentiment label) and 'text' (review content). The `prepare_data` function in `model/utils.py` handles the following preprocessing steps:

1. Tokenization of review text using the BERT tokenizer.
2. Padding/truncation of sequences to a fixed length (MAX_LENGTH in config.py).
3. Creation of attention masks.
4. Encoding of sentiment labels.

## Training Process

The training process (implemented in `model/train.py`) involves:

1. Loading and preprocessing the data.
2. Creating train and test DataLoader objects.
3. Initializing the model, loss function (CrossEntropyLoss), and optimizer (AdamW).
4. Training the model for a specified number of epochs.
5. Evaluating the model on the test set after each epoch.
6. Saving the best model weights.

## API and Web Interface

The project includes a FastAPI-based API and a simple web interface for interacting with the trained model:

- The API (`app/api/routes.py`) provides a `/predict` endpoint that accepts review text and returns the predicted sentiment.
- The web interface (`app/templates/index.html` and associated static files) allows users to input reviews and see the classification results.

## Setup and Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the model:

```bash
python3 scripts/train_model.py
```

3. Run the FastAPI application:

```bash
bash scripts/run_app.sh
```

4. Access the web interface at `http://localhost:8000` or make POST requests to `http://localhost:8000/predict` with JSON data in the format: `{"text": "Your review text here"}`.

## Customization

- To use a different dataset, update the file path in `model/config.py` and ensure it follows the expected CSV format.
- To modify the model architecture, edit `model/model.py`.
- To change hyperparameters, update `model/config.py`.

## Notes

- Ensure you have sufficient computational resources, especially if using GPU acceleration for training.
- The model weights file can be large. Make sure you have adequate storage space.
- For production deployment, consider using a production-grade server setup and implement appropriate security measures.

## Future Improvements

- Implement cross-validation for more robust evaluation.
- Experiment with different BERT variants or other transformer models.
- Add support for multi-language reviews.
- Implement active learning to continuously improve the model with user feedback.
- Explore transfer learning from other sentiment analysis tasks.
