import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense

# Load the IMDB dataset word index
word_index=imdb.get_word_index()
reversed_word_index = {value:key for key, value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper functions
# Function to decode reviews
def decode_review(encoded_review):
    return ''.join([reversed_word_index.get(i-3, '?') for i in encoded_review])

#Function to process user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen= 500)
    return padded_review

### Step 3:
### Prediction Function

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)

    prediction = model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

import streamlit as st
##streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

#User Input
user_input = st.text_area('Movie_Review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)

    ## Make prediction

    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    #Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')
