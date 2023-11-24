import numpy as np
import pandas as pd

import string
import re
import nltk
import pickle

from nltk.stem import PorterStemmer
ps = PorterStemmer()

# load model
with open('static/model/model.pickle', 'rb') as f:
    model = pickle.load(f)

# load stopwords
with open('static/model/corpora/stopwords/english', 'r') as file:
    sw = file.read().splitlines()

# load tokens
vocab = pd.read_csv('static/model/vocabulary.txt', header=None)
tokens = vocab[0].tolist()

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    text_without_punct = text.translate(translator)
    return text_without_punct

def remove_links(text):
    # Define a regular expression pattern to match URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    
    # Use sub() method to replace URLs with an empty string
    text_without_links = url_pattern.sub('', text)
    
    return text_without_links

def remove_numbers(text):
    # Define a regular expression pattern to match numbers
    number_pattern = re.compile(r'\d+')
    
    # Use sub() method to replace numbers with an empty string
    text_without_numbers = number_pattern.sub('', text)
    
    return text_without_numbers


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

def remove_stopwords(text):
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Remove stopwords
    filtered_words = [word for word in words if word.lower() not in stopwords.words('english')]
    
    # Join the filtered words back into a sentence
    text_without_stopwords = ' '.join(filtered_words)
    
    return text_without_stopwords


import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def apply_stemming(text):
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Initialize the Porter Stemmer
    porter_stemmer = PorterStemmer()
    
    # Apply stemming to each word
    stemmed_words = [porter_stemmer.stem(word) for word in words]
    
    # Join the stemmed words back into a sentence
    text_after_stemming = ' '.join(stemmed_words)
    
    return text_after_stemming


def preprocessing(text):
    data = pd.DataFrame([text], columns=['tweet'])
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(x.lower () for x in x.split()))
    data["tweet"] = data["tweet"].apply(remove_links)
    data["tweet"] = data["tweet"].apply(remove_punctuation)
    data["tweet"] = data["tweet"].apply(remove_numbers)
    data["tweet"] = data["tweet"].apply(remove_stopwords)
    data["tweet"] = data["tweet"].apply(apply_stemming)
    return data["tweet"]


def vectorizer(ds):
    vectorized_lst = []

    for sentence in ds:
        sentence_lst = np.zeros(len(tokens))

        for i in range(len(tokens)):
            if tokens[i] in sentence.split():
                sentence_lst[i] = 1

        vectorized_lst.append(sentence_lst)
        
    vectorized_lst_new = np.asarray(vectorized_lst, dtype = np.float32)
        
    return vectorized_lst_new



def get_prediction(vectorized_text):
    prediction = model.predict(vectorized_text)
    if prediction == 1:
        return 'negative'
    else:
        return 'positive'