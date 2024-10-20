import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pickle
import os
import streamlit as st

# load model
model = tf.keras.models.load_model('word_genearation_lstm.h5')

# load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# generate next word
def generate_next_word(model, tokenizer, input_text):
    max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list],maxlen = max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# streamlit app
st.title('Next Word Prediction')

input_text = st.text_input('Enter a sentence:')

if st.button('Predict next word'):
    next_word = generate_next_word(model, tokenizer, input_text)
else:
    next_word = None

if next_word:
    st.write('Next word:', next_word)
else:
    st.write('No next word found.')