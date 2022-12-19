from gensim.models import KeyedVectors 
import os
vector = KeyedVectors.load_word2vec_format(os.getcwd()+'/Word2Vec/vi.vec')