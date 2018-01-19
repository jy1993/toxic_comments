import fasttext
import pandas as pd
from sklearn.metrics import log_loss

# word reprsentation
# model = fasttext.skipgram('./data/data.txt', './model/skipgram')

# text classifier
classifier = fasttext.supervised('./data/train.data.txt', './model/classifier', 
	label_prefix='__label__')

print(classifier.predict(['example1 ']))
result = classifier.test('./data/val.data.txt')
print(result.precision)
print(result.recall)
print(result.nexamples)