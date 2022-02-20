"""a docstring"""
import random
from glob import glob
from functools import partial
import json
from collections import defaultdict
from os.path import exists
from gensim.utils import simple_preprocess
from gensim import corpora, models, similarities
import gensim.downloader as api
from gensim.similarities.annoy import AnnoyIndexer
import numpy

CACHE_DIR = "cache"
USE_GOOGLE = True


def get_corpus_and_dictionary_and_freshness():
    corpus = []
    for filename in glob("exports/*.content"):
        with open(filename, "r", encoding="utf8") as f:
            documents = json.load(f)
            corpus += [simple_preprocess(doc, deacc=True) for doc in documents]
    frequency: dict[str, int] = defaultdict(int)
    for text in corpus:
        for token in text:
            frequency[token] += 1
    trimmed_corpus = [[t for t in doc if frequency[t] > 1] for doc in corpus]
    dictionary = corpora.Dictionary(trimmed_corpus)
    dict_changed = False
    if exists(f"{CACHE_DIR}/corpus.dictionary"):
        dict_from_file = corpora.Dictionary.load(f"{CACHE_DIR}/corpus.dictionary")
        dict_changed = dict_from_file.token2id != dictionary.token2id
    if dict_changed:
        dictionary.save(f"{CACHE_DIR}/corpus.dictionary")

    return (
        trimmed_corpus,
        dictionary,
        dict_changed,
    )  # if dict is changed, I do want to remake


def create_or_load(name, creation_func, load_func, remake=False):
    if remake or not exists(f"{CACHE_DIR}/{name}"):
        obj = creation_func()
        obj.save(f"{CACHE_DIR}/{name}")
    else:
        obj = load_func(f"{CACHE_DIR}/{name}")
    return obj


def load_indexer(name):
    indexer = AnnoyIndexer()
    indexer.load(f"{CACHE_DIR}/{name}")
    return indexer


def create_word_vectors(corpus, name):
    model = models.Word2Vec(sentences=corpus, vector_size=20, epochs=100)
    word_vectors.save(f"{CACHE_DIR}/{name}")
    return word_vectors


def use_google_word_vectors(name):
    word_vectors = api.load("word2vec-google-news-300")
    word_vectors.save(f"{CACHE_DIR}/{name}")
    return word_vectors


def create_similarity_matrix(name, word_vectors, dictionary, tfidf):
    indexer = AnnoyIndexer(word_vectors, num_trees=2)
    termsim_index = similarities.WordEmbeddingSimilarityIndex(
        word_vectors, kwargs={"indexer": indexer}
    )
    similarity_matrix = similarities.SparseTermSimilarityMatrix(
        termsim_index, dictionary, tfidf
    )
    similarity_matrix.save(f"{CACHE_DIR}/{name}")
    return similarity_matrix


print("Importing Corpus")
corpus, dictionary, need_to_remake = get_corpus_and_dictionary_and_freshness()
if need_to_remake:
    print("looks like an updated dictionary!")
print("Creating Initial W2V Model")
if USE_GOOGLE:
    word_vectors_file_name = "google_word2vec.model"
    create = partial(use_google_word_vectors, word_vectors_file_name)
else:
    word_vectors_file_name = "word2vec.model"
    create = partial(create_word_vectors, corpus, word_vectors_file_name)
word_vectors = create_or_load(
    word_vectors_file_name,
    creation_func=create,
    load_func=models.KeyedVectors.load,
    remake=need_to_remake,
)
# print("Creating TFIDF")
# tfidf = create_or_load(
#     "tfidf.model",
#     creation_func=partial(models.TfidfModel, dictionary=dictionary),
#     load_func=models.TfidfModel.load,
#     remake=need_to_remake,
# )
# tfidf_corpus = tfidf[[dictionary.doc2bow(doc) for doc in corpus]]

# print("Creating indexers")
# similarity_matrix_file_name = "similarity.matrix"
# similarity_matrix = create_or_load(
#     similarity_matrix_file_name,
#     creation_func=partial(
#         create_similarity_matrix,
#         similarity_matrix_file_name,
#         word_vectors,
#         dictionary,
#         tfidf,
#     ),
#     load_func=similarities.SparseTermSimilarityMatrix.load,
#     remake=need_to_remake,
# )


def similarity_formatter(raw_sim):
    if type(raw_sim) == float or type(raw_sim) == numpy.float32:
        return f"{100*raw_sim:4.2f}%"
    return raw_sim


def play(dictionary, word_vectors):
    chosen = False
    while not chosen:
        secret_word = random.choice(list(dictionary.values()))
        try:
            word_vectors.similarity(secret_word, secret_word)
            chosen = True
        except:
            pass

    most_similar = word_vectors.similar_by_word(secret_word)
    print(
        f"The most similar word is {similarity_formatter(most_similar[0][1])} similar to the secret word"
    )

    while 1:
        w1 = input("Guess: ")
        if w1 == "quit":
            print(f"Secret word was {secret_word}")
            break
        if w1 == secret_word:
            print("You guessed it!")
        good_sim = f"I dont know {w1}"
        try:
            good_sim = word_vectors.similarity(secret_word, w1)
        except Exception:
            pass
        print(f"similarity={similarity_formatter(good_sim)}")


play(dictionary, word_vectors)
