import pickle
import util
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
import joblib

def tfidf_lr_main(col='toxic'):
	print('loading data')
	df_train = pd.read_csv(util.train_data)
	X = pickle.load(open(util.tmp_tf_idf_train, 'rb'))
	y = df_train[col]

	X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)
	print('X_train shape: {}'.format(X_train.shape))
	print('X_eval shape: {}'.format(X_eval.shape))

	print('trainig lr model for {}...'.format(col))
	lr = LogisticRegression()
	lr.fit(X_train, y_train)

	print('evaluating')
	y_pred = lr.predict_proba(X_eval)[:, 1]
	auc = roc_auc_score(y_eval, y_pred)
	logloss = log_loss(y_eval, y_pred)
	print('auc: {}, logloss: {}'.format(auc, logloss))

	print('saving model')
	joblib.dump(lr, util.lr_ver0 + '_' + col + '.pkl')

if __name__ == '__main__':
	for col in util.cols:
		tfidf_lr_main(col)