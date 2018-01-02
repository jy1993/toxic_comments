import pandas as pd
import util
from keras.preprocessing import text, sequence
import pickle
import numpy as np

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
corpus_seq = list(map(len, map(text.text_to_word_sequence, corpus)))
print(np.percentile(corpus_seq, [0, 25, 50, 75, 80, 90, 95, 100]))
# exit(2)

tokenizer = text.Tokenizer(num_words=util.num_words)
tokenizer.fit_on_texts(corpus)

print('texts to seqs...')
train_word_seqs = tokenizer.texts_to_sequences(train_comments)
test_word_seqs = tokenizer.texts_to_sequences(test_comments)

print('padding seqs...')
# train_padded_words_seqs = sequence.pad_sequences(train_word_seqs, maxlen=util.maxlen)
# test_padded_words_seqs = sequence.pad_sequences(test_word_seqs, maxlen=util.maxlen)

train_padded_words_seqs = sequence.pad_sequences(train_word_seqs, maxlen=util.maxlen_ver1)
test_padded_words_seqs = sequence.pad_sequences(test_word_seqs, maxlen=util.maxlen_ver1)

print('saving data...')
# pickle.dump(train_padded_words_seqs, open(util.tmp_padded_seq_train, 'wb'))
# pickle.dump(test_padded_words_seqs, open(util.tmp_padded_seq_test, 'wb'))

pickle.dump(train_padded_words_seqs, open(util.tmp_padded_seq_train_ver1, 'wb'))
pickle.dump(test_padded_words_seqs, open(util.tmp_padded_seq_test_ver1, 'wb'))


