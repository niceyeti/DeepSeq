"""
This files contains some brief word2vec/fasttext word vector model wrappers. Fasttext models are loaded via gensim using a deprecated
method; Word2Vec models using gensim.

Fasttext models: These must contain 'fasttext' in the file name to identify them.
IIRC, these models are the smaller ones (vectors only), with 'vec' somewhere in the filename, instead of 'bin', which is just a fasttext convention.

This file is duplicated in Sentinel.
"""

import gensim


#A wrapper to make fasttext KeyedVector model look like a gensim Word2Vec model
class FastTextModelWrapper(object):
	def __init__(self, model):
		self.wv = model
		self.vector_size = model.vector_size

"""
Fasttext models can be loaded into gensim, although it aint clear how long this will last, or fasttext will overtake gensim
From the developer:
	'For .bin use: load_fasttext_format() (this typically contains full model with parameters, ngrams, etc).
	For .vec use: load_word2vec_format (this contains ONLY word-vectors -> no ngrams + you can't update an model).
	Note:: If you are facing issues with the memory or you are not able to load .bin models, then check the pyfasttext model for the same.'

Also, gensim.models.wrappers.FastText.load_word2vec_format is deprecated, and it cannot continue to be trained upon (in gensim, at least; surely it
can in fasttext). But it also sounds like bullshit: if vectors (models) can be loaded, they can be trained upon.

"""
def loadFastTextModel(modelPath):
	#for .vec models only; see header comment
	if modelPath.endswith(".vec"):
		print("Loading fasttext model from {}... this can take up to fifteen minutes...".format(modelPath))
		print("If you are just testing/dev'ing, consider adding 'limit=' param to gensim.models.KeyedVectors.load_word2vec_format().")
		#model = gensim.models.wrappers.FastText.load_word2vec_format(modelPath, limit=100000)
		#model = gensim.models.KeyedVectors.load_word2vec_format(modelPath, limit=100000)
		model = gensim.models.KeyedVectors.load_word2vec_format(modelPath)
		return FastTextModelWrapper(model)
	else:
		print("ERROR fasttext model must end in .vec. See loadFasttestModel()")
		return None

def loadModel(modelPath):
	print("Loading vector model from: {}. Note fasttext models must contain 'fasttext' in filename.".format(modelPath))
	if "fasttext" in modelPath.lower():
		return loadFastTextModel(modelPath)
	return gensim.models.Word2Vec.load(modelPath)
