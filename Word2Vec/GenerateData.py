from .WordEncodings import WordEncodings
from .util import bytes_to_gb
from .UniGram import UniGram
from . import log

from tqdm import tqdm, trange
import torch

def encode_indexed_data_point(index, dtype, vocabulary_size):
    '''
    for every input X it returns an array of the vocabulary size where 1 of the
    values is 1 for the word it represents and the remaining values are 0

    @type x: [int]
    @param x: list of indexed training values
    @type vocabulary_size: int
    @param vocabulary_size: total number of words found
    '''
    indexed = torch.zeros(vocabulary_size, dtype=dtype)
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
    
    unigram = UniGram()

    encodings = WordEncodings(config, verbose=verbose)
    encodings.add_word(config.start_of_sentence_token)
    encodings.add_word(config.end_of_sentence_token)

    start_of_sentence_index = encodings.get_index_confident(config.start_of_sentence_token)
    end_of_sentence_index = encodings.get_index_confident(config.end_of_sentence_token)

    window_size = config.window_size

    for i in trange(len(tokenized_sentences), desc='building unigram '):
        sentence_length = len(tokenized_sentences[i])

        for j in range(sentence_length):
            word = tokenized_sentences[i][j]
            encodings.add_word(word)
            unigram.add_word(encodings.get_index_confident(word))

            tokenized_sentences[i][j] = word

    for sentence in tqdm(tokenized_sentences, desc='reading sentences'):
        sentence_length = len(sentence)

        for i in range(sentence_length):
            word = sentence[i]
            encodings.add_word(word)
            word_index = encodings.get_index_confident(word)

            if unigram.get_word_count(word_index) < config.minimum_count:
                continue

            for j in range(i - window_size, i + window_size + 1):
                if i == j:
                    continue

                if j < 0:
                    if config.use_start_and_end_tokens:
                        x.append(start_of_sentence_index)
                        y.append(torch.tensor([word_index], dtype=torch.long))
                elif j >= sentence_length:
                    if config.use_start_and_end_tokens:
                        x.append(end_of_sentence_index)
                        y.append(torch.tensor([word_index], dtype=torch.long))
                else:
                    encodings.add_word(sentence[j])
                    x.append(encodings.get_index_confident(sentence[j]))
                    y.append(torch.tensor([word_index], dtype=torch.long))
    
    return encodings, unigram, x, y
