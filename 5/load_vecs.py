from gensim.models import KeyedVectors
wv_from_text = KeyedVectors.load_word2vec_format('/Users/evdodima/education/fasttext_pmc.vec', binary=False)
print(wv_from_text.get_vector("man"))
