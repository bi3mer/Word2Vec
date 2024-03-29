# Word2Vec Skip-Grams

## TODO

* remove uncommon words
* Heiarachal softmax
* Move to numpy completely, pytorch isn't fast enough and GPU isn't really usable here
* Negative samples

## Samples

### Create a Small Model

```python
import Word2Vec

corpus = []
corpus.append('i imagine the day that I can run again , and it will be brilliant .'.split(' '))
corpus.append('the boy went to the store , and bought beef .'.split(' '))

config = Word2Vec.Config()

model = Word2Vec.SkipGram(config, corpus)
model.train()

Word2Vec.IO.save_embeddings(model, 'embeddings_dir')
Word2Vec.IO.save_model(model, 'model_dir')
```

### Load Saved Embeddings

```python
>>> import Word2Vec
>>> embeddings, encodings, config = Word2Vec.IO.load_embeddings('embeddings_dir')
>>> encodings.get_index('imagine')
4
```

### Load and Test Saved Model 

```python
>>> import numpy as np
>>> import Word2Vec
>>> model = Word2Vec.IO.load_model('model_dir')
>>> embeddings, encodings = Word2Vec.load_embeddings('embeddings_dir')
>>> np.array_equal(model.get_embedding('imagine'), embeddings['imagine'])
True
```

### K Nearest Neighbors

```python
>>> import numpy as np
>>> import Word2Vec
>>> embeddings, encodings, config = Word2Vec.IO.load_embeddings('embeddings_dir')
>>> neighbors = Word2Vec.NearestNeighbor(embeddings, encodings, config)
>>> neighbors.nearest_neighbors('day', count=3)
[(0.299554492104162, 'the'), (0.46842916371794735, 'imagine'), (0.5892674981281726, 'went')]
```

## Optimizations

Current Best on Brown Corpus: 354.38it/s

### Array Construction for Encodings

```python
import numpy as np
import time

iterations = 10000
array_size = 80000
index = 3

start = time.time()
for _ in range(iterations):
    a = [0 for i in range(array_size)]
    a[index] = 0
    a = np.array(a)

end = time.time()
print(f'Array to Numpy: {end - start}')

start = time.time()
for _ in range(iterations):
    a = np.zeros(array_size, dtype=int)
    a[index] = 1

end = time.time()
print(f'Numpy Zeroes: {end - start}')
```

```
Array to Numpy: 60.99235486984253
Numpy Zeroes: 0.1362009048461914
```

## Works Cited

* [Skip-Gram Paper](https://arxiv.org/pdf/1301.3781.pdf%5D)
* [Word2Vec Ppaer](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
* [Article I read to learn the concept of word2vec and skip-grams](https://towardsdatascience.com/learn-word2vec-by-implementing-it-in-tensorflow-45641adaf2ac)
* [My implementation reference](https://towardsdatascience.com/an-implementation-guide-to-word2vec-using-numpy-and-google-sheets-13445eebd281)
* [Optimization tips](https://rare-technologies.com/word2vec-in-python-part-two-optimizing/)