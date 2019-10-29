import pickle
import os

from .SkipGram import SkipGram
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
    config = load_pickle(os.path.join(directory_path, 'config.pkl'))

    return embeddings, encodings, config

def load_model(directory_path):
    if not os.path.isdir(directory_path):
        log.error(f'"{directory_path}" is not a directory.')
        return None, None

    encodings = load_pickle(os.path.join(directory_path, 'encoding.pkl'))
    config = load_pickle(os.path.join(directory_path, 'config.pkl'))
    w1 = load_pickle(os.path.join(directory_path, 'w1.pkl'))
    w2 = load_pickle(os.path.join(directory_path, 'w2.pkl'))

    model = SkipGram(config, encodings)
    model.w1 = w1
    model.w2 = w2

    return model

def save_embeddings(model, embeddings_directory):
    '''
    This will save the results of `get_all_embeddings`
    '''
    if not os.path.exists(embeddings_directory):
        log.warning(f'"{embeddings_directory}" directory does not exist. Making it now.')
        os.makedirs(embeddings_directory)

    f = open(os.path.join(embeddings_directory, 'embeddings.pkl'), 'wb')
    pickle.dump(model.get_all_embeddings(), f)
    f.close()

    f = open(os.path.join(embeddings_directory, 'encoding.pkl'), 'wb')
    pickle.dump(model.encodings, f)
    f.close()

    f = open(os.path.join(embeddings_directory, 'config.pkl'), 'wb')
    pickle.dump(model.config, f)
    f.close()

def save_model(model, model_directory):
    '''
    This will save the model and it's weights/configuration. Not the 
    embeddings directly.
    '''
    if not os.path.exists(model_directory):
        log.warning(f'"{model_directory}" directory does not exist. Making it now.')
        os.makedirs(model_directory)

    f = open(os.path.join(model_directory, 'w1.pkl'), 'wb')
    pickle.dump(model.w1, f)
    f.close()

    f = open(os.path.join(model_directory, 'w2.pkl'), 'wb')
    pickle.dump(model.w2, f)
    f.close()

    f = open(os.path.join(model_directory, 'encoding.pkl'), 'wb')
    pickle.dump(model.encodings, f)
    f.close()

    f = open(os.path.join(model_directory, 'config.pkl'), 'wb')
    pickle.dump(model.config, f)
    f.close()