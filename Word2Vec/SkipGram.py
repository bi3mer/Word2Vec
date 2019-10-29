from tqdm import trange
import numpy as np

import torch
from torch.autograd import Variable
from torch.nn.functional import log_softmax, nll_loss

from .GenerateData import encode_indexed_data_point
from .Config import Config
from . import log

class SkipGram():
    def __init__(self, config, encodings):
        self.encodings = encodings
        self.config = config

        self.vocabulary_size = encodings.vocabulary_size()

        self.w1 = Variable(torch.randn(config.encoding_size, self.vocabulary_size, dtype=self.config.dtype), requires_grad=True)
        self.w2 = Variable(torch.randn(self.vocabulary_size, config.encoding_size, dtype=self.config.dtype), requires_grad=True)
    
    def train(self, x, y):
        input_size = len(x)
        learning_rate = self.config.learning_rate
        dtype = self.config.dtype

        for epoch in range(self.config.epochs):
            correct_predictions = 0
            loss_value = 0
            
            for i in trange(input_size):
                x_vector = encode_indexed_data_point(x[i], dtype, self.vocabulary_size)
                target = y[i]

                output_1 = torch.matmul(self.w1, x_vector)
                output_2 = torch.matmul(self.w2, output_1)

                log_predictions = log_softmax(output_2, dim=0)
                loss = nll_loss(log_predictions.view(1,-1), target)
                loss_value += float(loss.data)
                loss.backward()

                self.w1.data -= learning_rate * self.w1.grad.data
                self.w2.data -= learning_rate * self.w2.grad.data

                self.w1.grad.data.zero_()
                self.w2.grad.data.zero_()

            print(f'Epoch {epoch} loss={loss_value/input_size}')

    def get_embedding(self, word):
        vector = encode_indexed_data_point(self.encodings.get_index(word), self.config.dtype, self.vocabulary_size)
        return torch.matmul(self.w1, vector).data

    def get_all_embeddings(self):
        '''
        for every word in the vocabulary, this will return the embedding with 
        word -> embedding
        '''
        word_to_vector = {}

        for word in self.encodings.words():
            word_to_vector[word] = self.get_embedding(word)

        return word_to_vector