# import nltk
# from nltk.corpus import stopwords

# def word_tokenize1(x):
#     lowerW = x.lower()
#     return nltk.word_tokenize(x)

# # read data ...
# # data =

# words = data.flatMap(word_tokenize1)
# print words.collect()

# stop_words=set(stopwords.words('english'))
# stopW = words.filter(lambda word : word[0] not in stop_words and word[0] != '')
# #print stopW.collect()

import time
input = "16.11.2019 14:25" # Hook['params']['date']
ts = time.strptime(input + " GMT+03:00", '%d.%m.%Y %H:%M GMT+03:00')
timestamp = time.mktime(ts)
print timestamp


