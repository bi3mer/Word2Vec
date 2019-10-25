from tqdm import tqdm
import numpy as np
import pickle
import os

from .Config import Config
from . import log

def load_pickle(file_path):
    try:
        f = open(file_path, 'rb')
        obj = pickle.load(f)
        f.close()

        return obj
    except IOError:
        log.error(f'"{file_path}" does not exist. Cannot be loaded.')
    except:
        log.error(f'Unable to read "{file_path}".')

    return None

def load_embeddings(directory_path):
    if not os.path.isdir(directory_path):
        log.error(f'"{directory_path}" is not a directory.')
        return None, None

    embeddings = load_pickle(os.path.join(directory_path, 'embeddings.pkl'))
    encodings = load_pickle(os.path.join(directory_path, 'encoding.pkl'))

    return embeddings, encodings

def load_model(directory_path):
    if not os.path.isdir(directory_path):
        log.error(f'"{directory_path}" is not a directory.')
        return None, None

    encodings = load_pickle(os.path.join(directory_path, 'encoding.pkl'))
    config = load_pickle(os.path.join(directory_path, 'config.pkl'))
    w1 = load_pickle(os.path.join(directory_path, 'w1.pkl'))
    w2 = load_pickle(os.path.join(directory_path, 'w2.pkl'))

    model = Model(config, encodings)
    model.w1 = w1
    model.w2 = w2

    return model

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

    def save_all_embeddings(self, embeddings_directory):
        '''
        This will save the results of `get_all_embeddings`
        '''
        if not os.path.exists(embeddings_directory):
            log.warning(f'"{embeddings_directory}" directory does not exist. Making it now.')
            os.makedirs(embeddings_directory)

        f = open(os.path.join(embeddings_directory, 'embeddings.pkl'), 'wb')
        pickle.dump(self.get_all_embeddings(), f)
        f.close()

        f = open(os.path.join(embeddings_directory, 'encoding.pkl'), 'wb')
        pickle.dump(self.encodings, f)
        f.close()

    def save_model(self, model_directory):
        '''
        This will save the model and it's weights/configuration. Not the 
        embeddings directly.
        '''
        if not os.path.exists(model_directory):
            log.warning(f'"{model_directory}" directory does not exist. Making it now.')
            os.makedirs(model_directory)

        f = open(os.path.join(model_directory, 'w1.pkl'), 'wb')
        pickle.dump(self.w1, f)
        f.close()

        f = open(os.path.join(model_directory, 'w2.pkl'), 'wb')
        pickle.dump(self.w2, f)
        f.close()

        f = open(os.path.join(model_directory, 'encoding.pkl'), 'wb')
        pickle.dump(self.encodings, f)
        f.close()

        f = open(os.path.join(model_directory, 'config.pkl'), 'wb')
        pickle.dump(self.config, f)
        f.close()