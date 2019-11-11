import nltk
from nltk.corpus import stopwords

def word_tokenize1(x):
    lowerW = x.lower()
    return nltk.word_tokenize(x)

# read data ...
# data =

words = data.flatMap(word_tokenize1)
print words.collect()

stop_words=set(stopwords.words('english'))
stopW = words.filter(lambda word : word[0] not in stop_words and word[0] != '')
#print stopW.collect()