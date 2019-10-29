from .WordEncodings import WordEncodings
from .util import bytes_to_gb
from . import log

from tqdm import tqdm
import torch

def encode_indexed_data_point(index, vocabulary_size):
    indexed = torch.zeros(vocabulary_size)
    indexed[index] = 1

    return indexed

def cuda_encode_indexed_data_point(index, vocabulary_size):
    indexed = torch.zeros(vocabulary_size, device='cuda')
    indexed[index] = 1

    return indexed



def generate_indexed_data(tokenized_sentences, config, verbose=True):
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
