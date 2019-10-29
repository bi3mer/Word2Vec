from . import log

class WordEncodings():
    def __init__(self, config, verbose=True):
        '''
        @type verbose: bool
        @param verbose: boolean for whether or not there will be log output on error.

        Unknown words are returned as 0
        '''
        self.verbose = verbose
        self.config = config

        self.word_to_int = {config.unknown_word: 0}
        self.int_to_word = [0]
        self.index = 1

    def word_in_vocab(self, word):
        return word in self.word_to_int

    def add_word(self, word):
        if word not in self.word_to_int:
            self.word_to_int[word] = self.index
            self.int_to_word.append(word)
            self.index += 1

    def get_word(self, index):
        if index >= self.index or index < 0:
            log.error('Index not within bounds of encodings. Returned "unknown".', verbose=self.verbose)

            return self.config.unknown_word

        print(index, self.index, len(self.int_to_word))
        return self.int_to_word[index]

    def get_index(self, word):
        if word not in self.word_to_int:
            log.error('Word not found. Returned 0.', verbose=self.verbose)

        return self.word_to_int[word]

    def get_index_confident(self, word):
        return self.word_to_int[word]

    def vocabulary_size(self):
        return self.index

    def indexes(self):
        return [i for i in range(self.index)]

    def words(self):
        return self.word_to_int.keys()