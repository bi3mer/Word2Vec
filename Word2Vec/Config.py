class Config():
    def __init__(self):
        '''
        Initialize configuration file which can and should be modified for more 
        involved processes.

        * window_size defines how many of the context words around the input word is used
        * encoding_size defines the size of the output embedding (hidden layer)
        * epochs defines how many epochs the word2vec model will run on
        * learning_rate defines the learning rate of the model
        * use_start_and_end_token defines whether start and end tokens will be used in the
          window size
        '''
        self.window_size = 3
        self.encoding_size = 10
        self.epochs = 10
        self.learning_rate = 0.01
        self.use_start_and_end_tokens = True

        self.start_of_sentence_token = '<start>'
        self.end_of_sentence_token = '<end>'
        self.unknown_word = '<UNK>'