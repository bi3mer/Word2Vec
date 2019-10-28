from tqdm import tqdm
import numpy as np
import pickle
import os

from .Config import Config
from . import log

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def heirarchal_softmax(x):
    raise NotImplementedError()

def forward(model, vector):
    hidden_layer = np.dot(model.w1.T, vector)
    u = np.dot(model.w2.T, hidden_layer)
    return hidden_layer, u, softmax(u)

def backwards_propagation(model, error, hidden_layer, vector):
    # TODO: I need to understand this better to come up with better variable names 
    #       and just for my general sanity
    dl_dw2 = np.outer(hidden_layer, error)
    dl_dw1 = np.outer(vector, np.dot(model.w2, error.T))

    model.w1 = model.w1 - (model.config.learning_rate * dl_dw1)
    model.w2 = model.w2 - (model.config.learning_rate * dl_dw2)

class SkipGram():
    def __init__(self, config, encodings):
        self.encodings = encodings
        self.config = config

        self.w1 = np.random.rand(encodings.vocabulary_size(), config.encoding_size)
        self.w2 = np.random.rand(config.encoding_size, encodings.vocabulary_size())
    
    def train(self, x, y):
        input_size = len(x)

        for epoch in range(self.config.epochs):
            correct_predictions = 0
            
            for i in range(input_size):
                vector = x[i]
                hidden_layer, u, predictions = forward(self, vector)

                error = np.sum([np.subtract(predictions, word) for word in vector], axis=0)
                backwards_propagation(self, error, hidden_layer, vector)

                if y[i] == np.argmax(predictions):
                    correct_predictions += 1

            print(f'Epoch {epoch} | Accuracy {correct_predictions / float(input_size)}')

    def get_embedding(self, word):
        return self.w1[self.encodings.get_index(word)]

    def get_all_embeddings(self):
        '''
        for every word in the vocabulary, this will return the embedding with 
        word -> embedding
        '''
        word_to_vector = {}

        for word in self.encodings.words():
            word_to_vector[word] = self.get_embedding(word)

        return word_to_vector