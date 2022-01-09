import streamlit as st
import numpy as np
import pandas as pd
from vncorenlp import VnCoreNLP
import joblib
import re
from text_utils import *
vncorenlp_file = './VnCoreNLP/VnCoreNLP-1.1.1.jar'

@st.cache(hash_funcs={VnCoreNLP: id})
def vncorenlp_setup():
    return VnCoreNLP(vncorenlp_file, annotators='wseg', max_heap_size='-Xmx500m')

annotator = vncorenlp_setup()

def text_tokenize(text):
    word_segmented_text = annotator.tokenize(text)
    new_text = '.'.join(' '.join(sentence) for sentence in word_segmented_text)
    return new_text

def text_preprocessing(text):
    text = remove_urls(text)
    text = remove_html(text)
    text = remove_numbers(text)
    text = text.lower()
    text = remove_punc(text)
    text = text_tokenize(text)
    text = remove_stopwords(text)
    return text


# ---- Naive Bayes model ----
def load_naive_bayes_model():
    return joblib.load('./models/multiNB_clf.sav')

multiNB_clf = load_naive_bayes_model()
def naive_bayes(text):
    return multiNB_clf.predict([text])

# ---- Logistic regression model ----
def load_logistic_model():
    return joblib.load('./models/lr_clf.sav')

lr_clf = load_logistic_model()
def logistic(text):
    return lr_clf.predict([text])

# ---- Steamlit ----

def predict(text, model):
    text = text_preprocessing(text)

    if model == 'Naive Bayes':
        m = naive_bayes
    else:
        m = logistic

    return f"Result: {'Fake news' if m(text)[0] == 1 else 'Real news' }"

st.title('Fake news detector')
st.header('Models')
st.selectbox('Choose model', ('Naive Bayes', 'Logistic Regression'), key='selectbox_model')
st.text_area('Insert a piece of news here', key='text_area')
if st.button('Predict', key='button_predict',):
    st.write(predict(st.session_state.text_area, st.session_state.selectbox_model))