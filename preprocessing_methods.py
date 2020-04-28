import numpy
import nltk
from nltk.stem import LancasterStemmer
import os
import tflearn
import tensorflow
import json
nltk.download('punkt')
stemmer = LancasterStemmer()

with open("tonydata.json") as file:
    data = json.load(file)


def extract_data():
    words, topics, lists_of_words, matched_topics = [], [], [], []

    for sub_data in data["data"]:
        for inp in sub_data["inputs"]:
            wrds = nltk.word_tokenize(inp)
            words.extend(wrds)
            lists_of_words.append(wrds)
            matched_topics.append(sub_data["topic"])

        if sub_data["topic"] not in topics:
            topics.append(sub_data["topic"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    topics = sorted(topics)
    return words, topics, lists_of_words, matched_topics


def get_training_data(words, topics, lists_of_words, matched_topics):
    training, output = [], []

    empty_output_matrix = [0 for _ in range(len(topics))]

    for x, list_of_word in enumerate(lists_of_words):
        training_matrix = []

        wrds = [stemmer.stem(w.lower()) for w in list_of_word]

        for w in words:
            if w in wrds:
                training_matrix.append(1)
            else:
                training_matrix.append(0)

        output_matrix = empty_output_matrix[:]
        output_matrix[topics.index(matched_topics[x])] = 1

        training.append(training_matrix)
        output.append(output_matrix)

    return numpy.array(training), numpy.array(output)


def develop_model(training, output):
    tensorflow.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 95)
    net = tflearn.fully_connected(net, 62)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    return tflearn.DNN(net)


def train_and_save(model, training, output):
    if os.path.exists("model.tflearn.meta"):
        model.load("model.tflearn")
    else:
        model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
        model.save("model.tflearn")


def get_binary_matrix(string, words):
    binary_matrix = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(string)
    s_words = [stemmer.stem(word.lower()) for word in s_words if word != "?"]

    for s_word in s_words:
        for i, word in enumerate(words):
            if s_word == word:
                binary_matrix[i] = 1

    return numpy.array(binary_matrix)

