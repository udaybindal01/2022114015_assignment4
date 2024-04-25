"""
Contains helper function to preprocess data and format it accordingly
"""

import csv
import string
from nltk.tokenize import word_tokenize

def file_read(file_path):

    """
    Takes file path as input
    Reads data from file and returns two lists, one of class labels and other of sentences.
    """

    classes = []
    sentences = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        i = 0
        for row in reader:
            if i == 0:
                i += 1
                continue
            classes.append(row[0])
            sentences.append(row[1])

    return classes, sentences

def clean_sentences(sentences):

    """
    Takes a list of sentences as input
    Returns a cleaned list of sentences (tokenization, removal of punctuation, lower case, etc)
    """

    cleaned_sentences = []
    for sentence in sentences:
        cleaned_tokens = [token for token in word_tokenize(sentence.lower()) if token not in string.punctuation]
        cleaned_sentence = ' '.join(cleaned_tokens)
        cleaned_sentences.append(cleaned_sentence)
    return cleaned_sentences

def make_vocab(sentences):

    """
    Takes a list of sentences as input
    Returns a vocabulary created out of those input sentences.
    """

    vocabulary = set()
    for sentence in sentences:
        words = word_tokenize(sentence)
        vocabulary.update(words)
    sorted_vocabulary = sorted(vocabulary)
    sorted_vocabulary.insert(0, "<UNK>")
    sorted_vocabulary.insert(1, "<BOS>")
    sorted_vocabulary.insert(2, "<EOS>")
    sorted_vocabulary.insert(3, "<PAD>")
    return sorted_vocabulary

def make_w_2_i(vocabulary):

    """
    Accepts the Vocabulary (set of words)
    Stores a word_to_index.json file
    """

    word_to_index = {word: index for index, word in enumerate(vocabulary)}
    return word_to_index

def main():

    _, train_sentences = file_read("try.csv")
    print(len(train_sentences))
    cleaned_train_sentences = clean_sentences(train_sentences)
    print(len(cleaned_train_sentences))
    vocabulary = make_vocab(cleaned_train_sentences)
    print(len(vocabulary))

if __name__ == '__main__':
    main()