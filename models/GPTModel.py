# In this code we have used GPT-2 Model for classification of IMDB Dataset:

# Code is as followed:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Config
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import re


# Defining cleaning function:
def clean(text):
    for token in ["<br/>", "<br>", "<br>"]:
        text = re.sub(token, " ", text)
    
    text = re.sub("[\s+\.\!\/_,$%^*()\(\)<>+\"\[\]\-\?;:\'{}`]+|[+——！，。？、~@#￥%……&*（）]+", " ", text)
    
    return text.lower()

# Loading Data:
def load_imdb_dataset(data_path, nrows=100):
    df = pd.read_csv(data_path, nrows=nrows)
    texts = df['review'].apply(clean)
    labels = df['sentiment']
    return texts, labels

# Class for GPT-Transformer
class GPT2Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer, max_length=2):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        input_ids = []
        for text in X:
            encoded_text = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_length, truncation=True)
            input_ids.append(encoded_text)
        return input_ids

# Class for the classifier
class GPT2Classifier(BaseEstimator):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self

    def predict(self, X):
        # Finding the maximum sequence length
        max_length = max(len(seq) for seq in X)

        # Pading sequences to the maximum length
        padded_input_ids = [seq + [0] * (max_length - len(seq)) for seq in X]

        # Convert input to tensors
        input_ids = torch.tensor(padded_input_ids)

        # Move input tensors to the appropriate device (GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = input_ids.to(device)

        # Moving the model to the appropriate device
        self.model.to(device)

        # Predicting logits
        with torch.no_grad():
            logits = self.model(input_ids)[0]

        # Moving logits back to CPU if necessary
        logits = logits.cpu()

        # Converting logits to class labels
        predicted_labels = torch.argmax(logits, dim=1).tolist()

        # Converting predicted labels to original label format
        label_map = {1: 'positive', 0: 'negative'}
        predicted_labels = [label_map[label] for label in predicted_labels]

        return predicted_labels

def main():
    data_path = "/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv"
    texts, labels = load_imdb_dataset(data_path, nrows=20000)  # Load only the top 100 rows
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Initializing tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Loading GPT2 configuration
    config = GPT2Config.from_pretrained('gpt2')
    config.pad_token_id = config.eos_token_id  # Set the padding token ID to the end-of-sequence token ID
    
    # Initializing model with updated configuration
    model = GPT2ForSequenceClassification.from_pretrained('gpt2', config=config)

    # Defining the pipeline for text classification
    pipeline = Pipeline([
        ('transformer', GPT2Transformer(tokenizer, max_length=2)),
        ('clf', GPT2Classifier(model)),
    ])

    # Training the classifier
    pipeline.fit(train_texts, train_labels)

    # Predicting on the test set
    predicted_labels = pipeline.predict(test_texts)

    # Calculating accuracy
    accuracy = accuracy_score(test_labels, predicted_labels)
    print("GPT2 Accuracy:", accuracy)

if __name__ == "__main__":
    main()


# Got Accuracy of 0.48825.
