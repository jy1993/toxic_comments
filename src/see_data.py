import pandas as pd
import collections
import util
import tensorflow as tf
from keras.preprocessing import text, sequence
import numpy as np

print('loading data')
df = pd.read_csv(util.train_data)
print(df.shape)
train_comments = df['comment_text'].tolist()
y_toxic = df['toxic']
print(df.columns)
for col in ['toxic', 'severe_toxic', 'obscene', 'threat',\
       'insult', 'identity_hate']:
    print('{0}, {1:.6f}'.format(col, df[col].mean()))
exit(1)

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(train_comments[:1])
sample_comment_in_seq = tokenizer.texts_to_sequences(train_comments[:1])
print(tokenizer.char_level)
print(tokenizer.word_counts)
print(tokenizer.word_index)
print(tokenizer.word_index['nonsense'])
print(train_comments[:1])
print(text.text_to_word_sequence(train_comments[0]))
print(sample_comment_in_seq)

print(sequence.pad_sequences(sample_comment_in_seq, maxlen=util.maxlen))

# comment_seq = map(text.text_to_word_sequence, train_comment[:10])
# print(list(comment_seq))