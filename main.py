import random
import pickle
from preprocessing_methods import *

try:
    with open("data.pickle", "rb") as f:
        words, topics, training, output = pickle.load(f)
except EnvironmentError:
    words, topics, lists_of_words, matched_topics = extract_data()
    training, output = get_training_data(words, topics, lists_of_words, matched_topics)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, topics, training, output), f)

model = develop_model(training, output)
train_and_save(model, training, output)


def main():
    print("==== I'm TonyBot! Ask away! Open-ended questions only! Type \"quit\" for me to leave ====")
    while True:
        user_inp = input("You: ")
        if user_inp == "quit":
            print("==== *TonyBot leaves* ====")
            break

        results = model.predict([get_binary_matrix(user_inp, words)])[0]
        results_index = numpy.argmax(results).item()
        topic = topics[results_index]

        if results[results_index] > 0.7:
            for tg in data["data"]:
                if tg["topic"] == topic:
                    print("TonyBot: " + random.choice(tg["replies"]))
        else:
            print("TonyBot: I don't understand, type something else.")


if __name__ == '__main__':
    main()
