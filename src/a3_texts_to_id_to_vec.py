import pandas as pd
import util
from keras.preprocessing import text, sequence
import pickle
import numpy as np
import os

print('loading data...')
df_train = pd.read_csv(util.train_data)
df_test = pd.read_csv(util.test_data)
df_train['comment_text'] = df_train['comment_text'].fillna('UN')
df_test['comment_text'] = df_test['comment_text'].fillna('UN')
print('df_test shape: {0}'.format(df_test.shape))

train_comments = df_train['comment_text'].tolist()
test_comments = df_test['comment_text'].tolist()

corpus = train_comments + test_comments
print('corpus size: {0}'.format(len(corpus)))

tokenizer = text.Tokenizer(num_words=util.num_words)
tokenizer.fit_on_texts(corpus)

print('Indexing word vectors.')
embeddings_index = {}
f = open(os.path.join(util.GLOVE_DIR, 'glove.6B.100d.txt'), 'rb')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

word_index = tokenizer.word_index

print('texts to seqs...')
train_word_seqs = tokenizer.texts_to_sequences(train_comments)
test_word_seqs = tokenizer.texts_to_sequences(test_comments)

print('padding seqs...')
train_padded_words_seqs = sequence.pad_sequences(train_word_seqs, maxlen=util.maxlen_ver1)
test_padded_words_seqs = sequence.pad_sequences(test_word_seqs, maxlen=util.maxlen_ver1)
print(train_padded_words_seqs.shape)

print('saving data...')
pickle.dump(train_padded_words_seqs, open(util.tmp_padded_seq_train_ver1, 'wb'))
pickle.dump(test_padded_words_seqs, open(util.tmp_padded_seq_test_ver1, 'wb'))

# prepare embedding matrix
num_words = min(util.num_words, len(word_index))
embedding_matrix = np.zeros((num_words + 1, util.EMBEDDING_DIM))
for word, i in word_index.items():
	if i >= util.num_words:
		continue
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		# words not found in embedding index will be all-zeros.
		embedding_matrix[i] = embedding_vector

print('saving embedding matrix')
pickle.dump(embedding_matrix, open(util.tmp_embedding_matrix, 'wb'))


