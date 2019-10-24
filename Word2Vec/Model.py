from tqdm import tqdm
import numpy as np
from . import log

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

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

class Model():
    def __init__(self, config, encodings, weights=None):
        self.encodings = encodings
        self.config = config

        if weights == None:
            self.w1 = np.random.rand(encodings.vocabulary_size(), config.encoding_size)
            self.w2 = np.random.rand(config.encoding_size, encodings.vocabulary_size())
        else:
            log.error('Loading past models is not supported... yet.')
    
    def train(self, x, y):
        input_size = len(x)

        for epoch in range(self.config.epochs):
            correct_predictions = 0
            
            for i in range(input_size):
                vector = x[i]
                hidden_layer, u, predictions = forward(self, vector)

                # TODO: test if np.array is faster
                error = np.sum([np.subtract(predictions, word) for word in vector], axis=0)
                backwards_propagation(self, error, hidden_layer, vector)

                if y[i] == np.argmax(predictions):
                    correct_predictions += 1

            print(f'Epoch {epoch} | Accuracy {correct_predictions / float(input_size)}')