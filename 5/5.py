from gensim.models import FastText
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors


import pickle
import logging

import os 

import json
import numpy as np
import pandas as pd




input_path = "/Users/evdodima/education/input_for_5/"

vec_path = "/Users/evdodima/education/fasttext_pmc.vec"
wv = KeyedVectors.load_word2vec_format(vec_path, binary=False)
print("слов в модели: " + str(len(wv.vocab)))

for train_filename in sorted(os.listdir(input_path)):
	input_data = []
	print(train_filename)
	if train_filename.endswith(".txt"):
		with open(input_path + train_filename, 'r') as train_file:
			for json_line in train_file:
				element = json.loads(json_line)
				input_data.append((element["label"],element["middle_context"]))

		print(input_data[:10])


		avg_vector = np.array([])
		words_number = 0

		vectors = []
		labels = np.array([])

		all_data = len(input_data)
		found_words = 0
		all_words = 0
		annotations = len(input_data)
		annotations_char_len = 0

		for context_tuple in input_data:
			for word in context_tuple[1]:
				all_words += 1
				annotations_char_len += len(word)
				try:
					v = np.array(wv.get_vector(word))
					if len(avg_vector) == 0:
						avg_vector = v
					else:
						avg_vector += v
					words_number += 1
					found_words += 1
				except KeyError:
					continue
					# print(word +" no such word")

			avg_vector = avg_vector / words_number
			vectors += [avg_vector]
			labels = np.append(labels, context_tuple[0])
			
		print(vectors[:10])
		print("найдено слов: " + str(found_words))
		print("всего слов: " + str(all_words))
		print("число аннотаций: " + str(annotations))
		print("средняя длина анотаций(слова): " + str(int(round(float(all_words)/float(annotations)))))
		print("средняя длина анотаций(символы): " + str(int(round(float(annotations_char_len)/float(annotations)))))


		result = pd.DataFrame()
		result["label"] = labels
		result["vector"] = vectors

		output = "/Users/evdodima/education/pkls/" + train_filename[:-4] + ".pkl"
		result.to_pickle(output)




# logging.basicConfig()
# logging.getLogger().setLevel(logging.INFO)

# for model_path in sorted(os.listdir("")):
# 	if model_path.endswith(".model"):
# 		model = FastText.load(fname)
