import pickle
import sys
from keras.layers import Dense, Dropout, LSTM, Embedding, Input
from keras.models import Model, save_model
from keras.callbacks import TensorBoard
import pandas as pd
import util

def pre_lstm_main(col='toxic', epochs=2, ver=0, mode='val'):
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

	embedding_matrix = pickle.load(open(util.tmp_embedding_matrix, 'rb'))
	embedding_layer = Embedding(util.num_words + 1,
								util.EMBEDDING_DIM,
								weights=[embedding_matrix],
								input_length=util.maxlen_ver1,
								trainable=False,
								mask_zero=True)

	# train a 1D convnet with global maxpooling
	sequence_input = Input(shape=(util.maxlen_ver1,), dtype='int32')
	embedded_sequences = embedding_layer(sequence_input)
	x = LSTM(rnn_units, dropout=dropout, recurrent_dropout=recurrent_dropout)(embedded_sequences)
	preds = Dense(1, activation='sigmoid')(x)

	model = Model(sequence_input, preds)

	model.compile(loss='binary_crossentropy',
				  optimizer='rmsprop',
				  metrics=['accuracy'])

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
		pre_lstm_main(col)
