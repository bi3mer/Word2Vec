from . import log

class WordEncodings():
    def __init__(self, verbose=True):
        '''
        @type verbose: bool
        @param verbose: boolean for whether or not there will be log output on error.

        Unknown words are returned as 0
        '''
        self.verbose = verbose

        self.word_to_int = {'unknown': 0}
        self.int_to_word = [0]
        self.index = 1

    def add_word(self, word):
        if word not in self.word_to_int:
            self.word_to_int[word] = self.index
            self.int_to_word.append(word)
            self.index += 1

    def get_word(self, index):
        if index >= self.index or index < 0:
            log.error('Index not within bounds of encodings. Returned "unknown".', verbose=self.verbose)

            return 'unknown'

        print(index, self.index, len(self.int_to_word))
        return self.int_to_word[index]

    def get_index(self, word):
        if word not in self.word_to_int:
            log.error('Word not found. Returned 0.', verbose=self.verbose)

        return self.word_to_int[word]