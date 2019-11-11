import nltk
from nltk.corpus import stopwords

def word_tokenize1(x):
    lowerW = x.lower()
    return nltk.word_tokenize(x)