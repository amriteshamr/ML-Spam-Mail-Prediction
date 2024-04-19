import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data from a CSV file
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['text'] = df['text'].apply(lambda x: x.replace('\r\n', ' '))
    return df

# Preprocess text data
def preprocess_text(df):
    stemmer = PorterStemmer()
    stopwords_set = set(stopwords.words('english'))
    corpus = []
    
    for text in df['text']:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation)).split()
        text = [stemmer.stem(word) for word in text if word not in stopwords_set]
        text = ' '.join(text)
        corpus.append(text)
    
    return corpus

# Train the model
def train_model(x_train, y_train):
    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(x_train, y_train)
    return clf

# Evaluate the model
def evaluate_model(clf, x_test, y_test):
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Make a prediction on a new email text
def classify_email(clf, vectorizer, email_text):
    email_text = email_text.lower()
    email_text = email_text.translate(str.maketrans('', '', string.punctuation)).split()
    
    stemmer = PorterStemmer()
    stopwords_set = set(stopwords.words('english'))
    email_text = [stemmer.stem(word) for word in email_text if word not in stopwords_set]
    email_text = ' '.join(email_text)
    
    email_corpus = [email_text]
    x_email = vectorizer.transform(email_corpus)
    prediction = clf.predict(x_email)
    
    return prediction

# Main function to run the script
def main():
    # Load the data
    filepath = 'spam_ham_dataset.csv'
    df = load_data(filepath)
    
    # Preprocess the text
    corpus = preprocess_text(df)
    
    # Vectorize the text
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(corpus).toarray()
    y = df['label_num']
    
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Train the model
    clf = train_model(x_train, y_train)
    
    # Evaluate the model
    accuracy = evaluate_model(clf, x_test, y_test)
    print(f'Model accuracy: {accuracy * 100:.2f}%')
    
    # Make a prediction on a sample email
    email_to_classify = df['text'].values[3]
    prediction = classify_email(clf, vectorizer, email_to_classify)
    
    # Print the prediction
    print(f'Prediction for email: {prediction}')

if __name__ == "__main__":
    main()
