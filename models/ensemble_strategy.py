# This file contains the ensemble methods like Random Forest, Gradient Boosting, AdaBoost Accuracy and Bagging Accuracy
# This has been applied on the data of IMDB.
# Following code would include whole process:



# Importing Neccessary Modules:
import re
import pandas as pd
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score
import re



# Data Preprocessing Function:

def clean(text):
    for token in ["<br/>", "<br>", "<br"]:
        text = re.sub(token, " ", text)

    text = re.sub("[\s+\.\!\/_,$%^*()\(\)<>+\"\[\]\-\?;:\'{}`]+|[+——！，。？、~@#￥%……&*（）]+", " ", text)

    return text.lower()


# Data Loading Function:

def load_imdb_dataset(data_path):
    df = pd.read_csv(data_path)

    texts = df['review'].apply(clean)
    labels = df['sentiment']



# Main function to load data and implementation of ensemble Methods:


def main():
    data_path = "/content/drive/MyDrive/DATASETS/IMDB Dataset.csv"

    texts, labels = load_imdb_dataset(data_path)

    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Defining classifiers for each ensemble method
    classifiers = {
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Bagging": BaggingClassifier(),
    }


    for name, classifier in classifiers.items():
        # Defining the pipeline for text classification
        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', classifier),
        ])

        
        pipeline.fit(train_texts, train_labels)

        
        predicted_labels = pipeline.predict(test_texts)

        # Calculating accuracy
        accuracy = accuracy_score(test_labels, predicted_labels)
        print(f"{name} Accuracy:", accuracy)

if __name__ == "__main__":
    main()



# Random Forest Accuracy: 0.8457
# Gradient Boosting Accuracy: 0.8162
# AdaBoost Accuracy: 0.807
# Bagging Accuracy: 0.7833