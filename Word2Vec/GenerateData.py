from .WordEncodings import WordEncodings

def generate_encoded_data(tokenized_sentences, config, verbose=True):
    '''
    generates encoded data set for training based on tokenized sentences

    @type tokenized_sentences: [[str]]
    @param tokenized_sentences: list of tokenized sentences from corpus
    @type config: Config
    @param config: defined configuration file
    @rtype: [[[int]], [int]]
    '''
    x = []
    y = []
    
    encodings = WordEncodings(verbose=verbose)
    encodings.add_word(config.start_of_sentence_token)
    encodings.add_word(config.end_of_sentence_token)

    start_of_sentence_index = encodings.get_index(config.start_of_sentence_token)
    end_of_sentence_index = encodings.get_index(config.end_of_sentence_token)

    window_size = config.window_size

    for sentence in tokenized_sentences:
        sentence_length = len(sentence)

        for i in range(sentence_length):
            word = sentence[i]
            encodings.add_word(word)
            word_index = encodings.get_index(word)

            for j in range(i - window_size, i + window_size + 1):
                if i == j:
                    continue

                if j < 0:
                    if config.use_start_and_end_tokens:
                        x.append(start_of_sentence_index)
                        y.append(word_index)
                elif j >= sentence_length:
                    if config.use_start_and_end_tokens:
                        x.append(end_of_sentence_index)
                        y.append(word_index)
                else:
                    encodings.add_word(sentence[j])
                    x.append(encodings.get_index(sentence[j]))
                    y.append(word_index)

    return encodings, x, y