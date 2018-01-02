import pickle
import sys
from keras.layers import Dense, Dropout, LSTM, Embedding, Input, Bidirectional
from keras.models import Model, save_model, Sequential
from keras.callbacks import TensorBoard
from keras import regularizers
import pandas as pd
import util

def stacked_bi_lstm_main(col='toxic', epochs=2, ver=1, mode='val', l2_value=0.01):
	#hyper paremeters for the lstm model
	embedding_len = 128

	batch_size = 32
	rnn_units = 64
	dropout = 0.2
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
	model.add(Bidirectional(LSTM(rnn_units, dropout=dropout, recurrent_dropout=recurrent_dropout, \
								 return_sequences=True)))
	model.add(LSTM(rnn_units, dropout=dropout, recurrent_dropout=recurrent_dropout))
	model.add(Dense(1, activation='sigmoid', activity_regularizer=regularizers.l2(l2_value)))

	print(model.summary())
	model.compile(loss='binary_crossentropy',
				  optimizer='rmsprop',
				  metrics=['accuracy'])

	print('trainging lstm model for %s...' % col)
	if mode == 'val':
		tb = TensorBoard(log_dir=util.lstm_ver0_log, histogram_freq=1)
		model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=2, \
				  validation_split=0.2, callbacks=[tb])
	#model.fit(x_train, y_train, batch_size=util.batch_size, epochs=epochs, verbose=1)
	elif mode == 'cv':
		model, _metrics = util.get_cv_logloss(model, X, y)

	print('saving model')
	if ver == 1:
		saving_path = util.stacked_bi_lstm_ver1 + '_' + col
	else:
		saving_path = util.stacked_bi_lstm_ver0 + '_' + col
	save_model(model, saving_path)

if __name__ == '__main__':
	for col in util.cols:
		stacked_bi_lstm_main(col)
