import pandas as pd
import spacy
import pickle
from keras.preprocessing import text
import numpy as np

print('loading data...')
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')
df_train['comment_text'] = df_train['comment_text'].fillna('UN')
df_test['comment_text'] = df_test['comment_text'].fillna('UN')
print('df_test shape: {0}'.format(df_test.shape))

train_comments = df_train['comment_text'].tolist()
test_comments = df_test['comment_text'].tolist()

corpus = train_comments + test_comments
print('corpus size: {0}'.format(len(corpus)))

# tokenizer = spacy.load('en')
tokenizer = text.text_to_word_sequence
# texts = list(map(tokenizer, corpus))
# print(texts[:3])

# with open('../data/data.txt', 'w', encoding='utf-8') as f:
# 	for text in texts:
# 		f.write(' '.join(text) + '\n')

# generating train, val, test data for fasttext classifier
np.random.seed(42)
index_array = np.arange(len(train_comments))
np.random.shuffle(index_array)
print(index_array)
train_comments = df_train['comment_text'].values
train_comments = train_comments[index_array]
y_train = df_train['toxic'].tolist()

print('splitting data...')
ratio = 0.2
_train = train_comments[:-int(ratio * len(train_comments))]
_val = train_comments[-int(ratio * len(train_comments)):]
_y_train = y_train[:-int(ratio * len(train_comments))]
_y_val = y_train[-int(ratio * len(train_comments)):]

train, train_label = df_train['comment_text'].values, df_train['toxic'].values
val, val_label = df_train['comment_text'].values, df_train['toxic'].values

with open('./data/train.data.txt', 'w', encoding='utf-8') as f:
	for i, text in enumerate(list(map(tokenizer, _train))):
		f.write('__label__{:d}\t'.format(_y_train[i]) + ' '.join(text) + '\n')

with open('./data/val.data.txt', 'w', encoding='utf-8') as f:
	for i, text in enumerate(list(map(tokenizer, _val))):
		f.write('__label__{:d}\t'.format(_y_val[i]) + ' '.join(text) + '\n')