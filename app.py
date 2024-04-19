from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the data and model
def load_data():
    df = pd.read_csv('spam_ham_dataset.csv')
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

# Load the data and preprocess
df = load_data()
corpus = preprocess_text(df)

# Vectorize the text
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(corpus).toarray()
y = df['label_num']

# Train the model
clf = RandomForestClassifier(n_jobs=-1)
clf.fit(x, y)

# Define the home page route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        email_text = request.form.get('email')
        
        # Preprocess the input email text
        stemmer = PorterStemmer()
        stopwords_set = set(stopwords.words('english'))
        email_text = email_text.lower()
        email_text = email_text.translate(str.maketrans('', '', string.punctuation)).split()
        email_text = [stemmer.stem(word) for word in email_text if word not in stopwords_set]
        email_text = ' '.join(email_text)
        
        # Convert the input email text to a vector
        x_email = vectorizer.transform([email_text])
        
        # Predict whether the email is spam or not
        prediction = clf.predict(x_email)
        
        # Return the result
        if prediction == 1:
            result = 'Spam'
        else:
            result = 'Not Spam'
        
        return render_template('index.html', result=result)
    
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
