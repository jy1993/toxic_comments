import pickle
import sys
from keras.layers import Dense, Dropout, Embedding, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.models import Sequential, save_model
import pandas as pd
import util

def cnn_main(col='toxic', epochs=2, ver=0, mode='val'):
	#hyper parameters for the cnn model
	filters = 250
	kernel_size = 3
	hidden_dims = 250
	embedding_dims = 50
	batch_size = 32

	#epochs = 2
	print('loading data')
	df_train = pd.read_csv(util.train_data)
	y = df_train[col].values
	if ver == 1:
		X = pickle.load(open(util.tmp_padded_seq_train, 'rb'))
	print('shape of X_train: {}'.format(X.shape))

	model = Sequential()
	model.add(Embedding(util.num_words, embedding_dims, input_length=util.maxlen))
	model.add(Dropout(0.2))
	#model.add(Masking())

	model.add(Conv1D(filters,
					 kernel_size,
					 padding='valid',
					 activation='relu',
					 strides=1))

	model.add(GlobalMaxPooling1D())

	model.add(Dense(hidden_dims))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))

	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])

	print('trainging cnn model for %s...' % col)
	if mode == 'val':
		model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=2, \
			  validation_split=0.1)
	else:
		model, _metrics = util.get_cv_logloss(model, X, y)
	#model.fit(x_train, y_train, batch_size=util.batch_size, epochs=epochs, verbose=1)

	print('saving model')
	if ver == 1:
		saving_path = util.cnn_ver1 + '_' + col
	else:
		saving_path = util.cnn_ver0 + '_' + col
	save_model(model, saving_path)

if __name__ == '__main__':
	for col in util.cols:
		cnn_main(col)