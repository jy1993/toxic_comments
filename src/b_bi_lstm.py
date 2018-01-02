import pickle
import sys
from keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from keras.models import Sequential, save_model
from keras.callbacks import TensorBoard
import pandas as pd
import util

def bi_lstm_main(col='toxic', epochs=3):
	#hyper paremeters for the lstm model
	embedding_len = 128
	#epochs = 1
	batch_size = 32
	rnn_units = 64
	dropout = 0.5

	print('loading data')
	df_train = pd.read_csv(util.train_data)
	y_train = df_train[col].tolist()
	x_train = pickle.load(open(util.tmp_padded_seq_train, 'rb'))
	print(x_train.shape, len(y_train))

	model = Sequential()
	model.add(Embedding(util.num_words, embedding_len, mask_zero=True))
	model.add(Bidirectional(LSTM(rnn_units)))
	model.add(Dropout(dropout))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])

	print('trainging bi_lstm model for %s...' % col)
	tb = TensorBoard(log_dir=util.bi_lstm_ver0_log, histogram_freq=1)
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, \
			  validation_split=0.1, callbacks=[tb])
	#model.fit(x_train, y_train, batch_size=util.batch_size, epochs=epochs, verbose=1)

	print('saving model')
	saving_path = util.bi_lstm_ver0 + '_' + col
	save_model(model, saving_path)

if __name__ == '__main__':
	for col in util.cols:
		bi_lstm_main(col)
