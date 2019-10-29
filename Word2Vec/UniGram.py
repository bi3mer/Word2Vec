class UniGram():
    def __init__(self):
        self.counts = []

    def add_word(self, word_index):
        while word_index >= len(self.counts):
            self.counts.append(0)

        self.counts[word_index] += 1

    def get_word_count(self, word_index):
        '''
        will throw error given bad word_index
        '''
        return self.counts[word_index]