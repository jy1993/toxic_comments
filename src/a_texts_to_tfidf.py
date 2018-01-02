from keras.preprocessing import text
import pandas as pd
import pickle
import util

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

tk = text.Tokenizer(num_words=1000)
tk.fit_on_texts(corpus)

tf_idf_train = tk.texts_to_matrix(train_comments, mode='tfidf')
tf_idf_test = tk.texts_to_matrix(test_comments, mode='tfidf')
print(tf_idf_train[:10])
pickle.dump(tf_idf_train, open(util.tmp_tf_idf_train, 'wb'))
pickle.dump(tf_idf_test, open(util.tmp_tf_idf_test, 'wb'))