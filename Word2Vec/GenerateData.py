from .WordEncodings import WordEncodings
from .util import bytes_to_gb
from . import log

from tqdm import tqdm
import torch

def encode_indexed_data_point(index, vocabulary_size):
    '''
    for every input X it returns an array of the vocabulary size where 1 of the
    values is 1 for the word it represents and the remaining values are 0

    @type x: [int]
    @param x: list of indexed training values
    @type vocabulary_size: int
    @param vocabulary_size: total number of words found
    '''
    indexed = torch.zeros(vocabulary_size)
    indexed[index] = 1

    return indexed

def generate_indexed_data(tokenized_sentences, config, verbose=True):
    '''
    generates encoded data set for training based on tokenized sentences

    @type tokenized_sentences: [[str]]
    @param tokenized_sentences: list of tokenized sentences from corpus
    @type config: Config
    @param config: defined configuration file
    @rtype: [[int]], [int]]
    '''
    x = []
    y = []
    
    encodings = WordEncodings(config, verbose=verbose)
    encodings.add_word(config.start_of_sentence_token)
    encodings.add_word(config.end_of_sentence_token)

    start_of_sentence_index = encodings.get_index_confident(config.start_of_sentence_token)
    end_of_sentence_index = encodings.get_index_confident(config.end_of_sentence_token)

    window_size = config.window_size

    for sentence in tqdm(tokenized_sentences, desc='reading sentences'):
        sentence_length = len(sentence)

        for i in range(sentence_length):
            word = sentence[i].lower()
            encodings.add_word(word)
            word_index = encodings.get_index_confident(word)

            for j in range(i - window_size, i + window_size + 1):
                if i == j:
                    continue

                if j < 0:
                    if config.use_start_and_end_tokens:
                        x.append(start_of_sentence_index)
                        y.append(torch.tensor([word_index]))
                elif j >= sentence_length:
                    if config.use_start_and_end_tokens:
                        x.append(end_of_sentence_index)
                        y.append(torch.tensor([word_index]))
                else:
                    encodings.add_word(sentence[j])
                    x.append(encodings.get_index_confident(sentence[j]))
                    y.append(torch.tensor([word_index]))
    
    return encodings, x, y
