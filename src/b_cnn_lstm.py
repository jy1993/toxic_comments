from keras.models import Sequential, save_model
from keras.layers import Embedding, Dropout, Dense
from keras.layers import Conv1D, MaxPooling1D, LSTM
from keras.callbacks import TensorBoard
import pandas as pd
import util
import pickle

def cnn_lstm_main(col='toxic', epochs=2, ver=0, mode='val'):
	embedding_dims = 128
	filters = 64
	kernel_size = 5
	lstm_output_size = 64
	batch_size = 32
	dropout = 0.5

	print('loading data')
	df_train = pd.read_csv(util.train_data)
	y = df_train[col].values
	if ver == 1:
		X = pickle.load(open(util.tmp_padded_seq_train_ver1, 'rb'))
	else:
		X = pickle.load(open(util.tmp_padded_seq_train, 'rb'))
	print(X.shape, y.shape)

	model = Sequential()
	model.add(Embedding(util.num_words, embedding_dims, input_length=util.maxlen))
	model.add(Dropout(dropout))
	model.add(Conv1D(filters, kernel_size, activation='relu'))
	model.add(MaxPooling1D())

	model.add(LSTM(lstm_output_size))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])

	print('trainging cnn_lstm model for %s...' % col)
	if mode == 'val':
		tb = TensorBoard(log_dir=util.cnn_lstm_ver0_log, histogram_freq=1)
		model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, \
			  validation_split=0.2, verbose=2, callbacks=[tb])
	else:
		model, _metrics = util.get_cv_logloss(model, X, y)

	print('saving model')
	if ver == 1:
		saving_path = util.cnn_lstm_ver1 + '_' + col
	else:
		saving_path = util.cnn_lstm_ver0 + '_' + col
	save_model(model, saving_path)