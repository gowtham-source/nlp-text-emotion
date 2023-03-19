import streamlit as st
import pickle

# text preprocessing
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re


def preprocess_and_tokenize(data):
    data = re.sub("(<.*?>)", "", data)
    data = re.sub(r'http\S+', '', data)
    data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)
    data = re.sub("(\\W|\\d)", " ", data)
    data = data.strip()
    data = word_tokenize(data)
    porter = PorterStemmer()
    stem_data = [porter.stem(word) for word in data]
    return stem_data


filename = 'tfidf_svm.sav'
model = pickle.load(open(filename, 'rb'))

st.write("## Text-Sentiment Analyzer")
message = st.text_area("Enter the Text")


# message = 'delivery was hour late and my pizza is cold!'
if st.button("Predict"):
    c = model.predict([message])
    st.write(f'#####     {c[0]}')
