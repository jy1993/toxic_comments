import pandas as pd
import util
import numpy as np
import argparse
from sklearn.model_selection import train_test_split


def main(config):
	col = config.col
	targets =  ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
	if col not in targets:
		print('expect one of the following cols: {}, got {}'.format(','.join(targets), col))
		raise ValueError
	print('loading data...')
	df_train = pd.read_csv(util.train_data)
	df_test = pd.read_csv(util.test_data)

	df_train['comment_text'] = df_train['comment_text'].fillna('UN')
	df_test['comment_text'] = df_test['comment_text'].fillna('UN')
	y_train = df_train[col].tolist()
	y_test = [0] * len(df_test['comment_text'])

	# print('shuffling data...')
	train_comments = df_train['comment_text'].values
	test_comments = df_test['comment_text'].values
	# index_array = np.arange(len(train_comments))
	# np.random.shuffle(index_array)
	# print(index_array)
	# train_comments = train_comments[index_array]

	# print('splitting data...')
	# ratio = 0.2
	# _train = train_comments[:-int(ratio * len(train_comments))]
	# _val = train_comments[-int(ratio * len(train_comments)):]
	# _test = test_comments

	# _y_train = y_train[:-int(ratio * len(train_comments))]
	# _y_val = y_train[-int(ratio * len(train_comments)):]
	# _y_test = y_test

	_train, _val, _y_train, _y_val = train_test_split(
		train_comments, y_train, test_size=0.2, random_state=42, stratify=y_train)
	_test = test_comments
	_y_test = y_test

	print('length of train: {}, length of val: {}, length of test: {}'.format(len(_train), len(_val), len(_test)))

	train = pd.DataFrame({'text': _train, 'label': _y_train})
	val = pd.DataFrame({'text': _val, 'label': _y_val})
	test = pd.DataFrame({'text': _test, 'label': _y_test})
	print(val.head())

	print('mean of train label: {}, mean of val label: {}'.format(train['label'].mean(), val['label'].mean()))
	print('saving data:')

	train.to_csv('../data/train_{}.csv'.format(col), index=False)
	val.to_csv('../data/val_{}.csv'.format(col), index=False)
	test.to_csv('../data/test_{}.csv'.format(col), index=False)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--col', type=str, default='toxic', 
		help='specify the target to predict')
	config = parser.parse_args()
	print(config)
	main(config)




