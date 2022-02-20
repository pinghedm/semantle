"""module docstring"""
import bottle
from bottle import response
from train_model import (
    get_corpus_and_dictionary_and_freshness,
    get_word_vectors,
    choose_secret_word,
    get_most_similar_word_similarity,
    similarity_formatter,
    get_similarity,
    UnknownWordException,
)
from bottle_cors_plugin import cors_plugin

# INIT
corpus, dictionary, need_to_remake = get_corpus_and_dictionary_and_freshness()
word_vectors = get_word_vectors(corpus, USE_GOOGLE=False, need_to_remake=False)
app = bottle.app()


# GAME ROUTES
@app.route("/initial_similarity/<seed:int>")
def get_initial_similarity(seed):
    secret_word = choose_secret_word(seed, dictionary, word_vectors)
    print(secret_word)
    most_similar_word_similarity = get_most_similar_word_similarity(
        secret_word, word_vectors
    )
    return {"similiarity": similarity_formatter(most_similar_word_similarity)}


@app.route("/guess/<seed:int>/<guess>")
def process_guess(seed, guess):
    secret_word = choose_secret_word(seed, dictionary, word_vectors)
    if guess == secret_word:
        return {
            "guess": guess,
            "rawSimilarity": 1.0,
            "similiarity": "100%",
            "known": True,
            "correctAnswer": True,
        }
    try:
        similiarity = get_similarity(secret_word, guess, word_vectors)
        known = True
    except UnknownWordException:
        similiarity = 0
        known = False
    return {
        "guess": guess,
        "rawSimilarity": float(similiarity),
        "similiarity": similarity_formatter(similiarity),
        "known": known,
        "correctAnswer": False,
    }


@app.route("/give_up/<seed:int>")
def give_up(seed):
    secret_word = choose_secret_word(seed, dictionary, word_vectors)
    return {"secretWord": secret_word}


app.install(cors_plugin("*"))
app.run(host="localhost", port=8000)
