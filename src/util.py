import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
import numpy as np

#data filepath
input_dir = '../input'
train_data = input_dir + '/train.csv'
test_data = input_dir + '/test.csv'

tmp_dir = '../tmp'
tmp_padded_seq_train = tmp_dir + '/tmp_padded_seq_train.pkl'
tmp_padded_seq_test = tmp_dir + '/tmp_padded_seq_test.pkl'

tmp_padded_seq_train_ver1 = tmp_dir + '/tmp_padded_seq_train_ver1.pkl'
tmp_padded_seq_test_ver1 = tmp_dir + '/tmp_padded_seq_test_ver1.pkl'

tmp_tf_idf_train = tmp_dir + '/tmp_tf_idf_train.pkl'
tmp_tf_idf_test = tmp_dir + '/tmp_tf_idf_test.pkl'

tmp_embedding_matrix = tmp_dir + './tmp_embedding_matrix.pkl'

model_dir = '../model'
lstm_ver0 = model_dir + '/lstm_ver0'
lstm_ver1 = model_dir + '/lstm_ver1'
bi_lstm_ver0 = model_dir + '/bi_lstm_ver0'
bi_lstm_ver1 = model_dir + '/bi_lstm_ver1'
cnn_ver0 = model_dir + '/cnn_ver0'
cnn_ver1 = model_dir + '/cnn_ver1'
lr_ver0 = model_dir + '/lr_ver0'
svm_ver0 = model_dir + 'svm_ver0'
cnn_lstm_ver0 = model_dir + '/cnn_lstm_ver0'
cnn_lstm_ver1 = model_dir + '/cnn_lstm_ver1'

stacked_bi_lstm_ver1 = model_dir + './stacked_bi_lstm_ver1'
stacked_bi_lstm_ver0 = model_dir + './stacked_bi_lstm_ver0'

log_dir = '../log'
bi_lstm_ver0_log = log_dir + '/bi_lstm_ver0'
lstm_ver0_log = log_dir + '/lstm_ver0'
cnn_lstm_ver0_log = log_dir + '/cnn_lstm_ver0'

output_dir = '../output'
output_lstm_ver0_pre = output_dir + '/lstm_ver0'
output_sub_lstm_ver0_pre = output_dir + '/sub_lstm_ver0'

#cols
cols = ['toxic', 'severe_toxic', 'obscene', 'threat', \
		'insult', 'identity_hate']

#hyper-parameters for the tensorflow model
# max_steps = 300
# n_input = 100
# n_neurons = 200
# n_layers = 3
# learning_rate = 0.001

#hyper-parameters for the keras model
#max number of words in embedding vocabulary
num_words = 20000
#max length of a comment before cutting off
maxlen = 100
maxlen_ver1 = 260

#word2vec file location
BASE_DIR = '../..'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
EMBEDDING_DIM = 100


def get_cv_logloss(model, X, y, n_fold = 3, metric = 'logloss', \
				   batch_size = 32, epochs = 2, \
				   random_state = 42):
	kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=random_state)
	_metrics = []
	for train_index, test_index in kfold.split(X, y):
		X_train, y_train = X[train_index], y[train_index]
		X_test, y_test = X[test_index], y[test_index]
		model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)
		y_pred = model.predict_proba(X_test, batch_size=batch_size, verbose=1)
		_metrics.append(log_loss(y_test, y_pred))
	return (model, _metrics)

def get_all_batches(X, y, batch_size, shuffle=True):
	'''
	get all the batches for X, y
	:param X: input features
	:param y: input labels
	:param batch_size:
	:param shuffle: whether to shuffle the data before get batches
	:return: data in batch
	'''
	assert X.shape[0] == y.shape[0]
	n_batches = X.shape[0] // batch_size
	if shuffle:
		indices = np.arange(X.shape[0])
		np.random.shuffle(indices)
		X_shuffled = X[indices]
		y_shuffled = y[indices]
		if debug:
			print(indices)
			print(X_shuffled)
			print(y_shuffled)
	else:
		X_shuffled = X
		y_shuffled = y

	X_to_return = [X_shuffled[i * batch_size: (i * batch_size + batch_size)] for i in range(n_batches)]
	y_to_return = [y_shuffled[i * batch_size: (i * batch_size + batch_size)] for i in range(n_batches)]
	return (X_to_return, y_to_return)

if __name__ == '__main__':
	'''
	testing util function
	'''
	debug = False
	X = np.arange(20).reshape((4, 5))
	y = np.array([0, 1, 2, 3])
	print(get_all_batches(X, y, 3))

