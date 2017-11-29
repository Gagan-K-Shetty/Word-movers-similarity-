from nltk.corpus import stopwords
from gensim.models import Word2Vec,KeyedVectors
from gensim.similarities import WmdSimilarity
from nltk import word_tokenize
stop_words = stopwords.words('english')

def preprocess(doc):
    doc = doc.lower()
    doc = word_tokenize(doc)
    doc = [w for w in doc if not w in stop_words]
    doc = [w for w in doc if w.isalpha()]
    return doc
text1="bad girl"
text2="good boy"
w2v_corpus = [preprocess(text1)]
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True,limit=5000)
model.init_sims(replace=True)
num_best = 1
instance = WmdSimilarity(w2v_corpus, model, num_best=num_best)
query=[preprocess(text2)]
sims = instance[query]
similarity=sims[0][1]
print("The sentences are ",similarity,"% similar")
