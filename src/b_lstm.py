import pickle
import sys
import random as rn
import numpy as np
import tensorflow as tf
import os
from keras import backend as K
import pandas as pd
import util

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/keras-team/keras/issues/2280#issuecomment-306959926

os.environ['PYTHONHASHSEED'] = '0'
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# Rest of code follows ...
from keras.layers import Dense, Dropout, LSTM, Embedding, Masking
from keras.models import Sequential, save_model
from keras.callbacks import TensorBoard

def lstm_main(col='toxic', epochs=2, ver=0, mode='val', dropout=0.2):
	#hyper paremeters for the lstm model
	embedding_len = 128

	batch_size = 32
	rnn_units = 64
	#dropout = 0.2
	#dropout = 0.5
	recurrent_dropout=0.2

	print('loading data')
	df_train = pd.read_csv(util.train_data)
	X = None
	y = df_train[col].values
	if ver == 1:
		X = pickle.load(open(util.tmp_padded_seq_train_ver1, 'rb'))
	else:
		X = pickle.load(open(util.tmp_padded_seq_train, 'rb'))
	print(X.shape, y.shape)

	model = Sequential()
	model.add(Embedding(util.num_words, embedding_len, mask_zero=True))
	model.add(LSTM(rnn_units, dropout=dropout, recurrent_dropout=recurrent_dropout))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

	print(model.summary())
	print('trainging lstm model for %s...' % col)
	if mode == 'val':
		tb = TensorBoard(log_dir=util.lstm_ver0_log, histogram_freq=1)
		model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=2, \
			validation_split=0.2, callbacks=[tb])
	#model.fit(x_train, y_train, batch_size=util.batch_size, epochs=epochs, verbose=1)
	else:
		model, _metrics = util.get_cv_logloss(model, X, y)

	print('saving model')
	if ver == 1:
		saving_path = util.lstm_ver1 + '_' + col
	else:
		saving_path = util.lstm_ver0 + '_' + col
	save_model(model, saving_path)

if __name__ == '__main__':
	for col in util.cols:
		lstm_main(col)
