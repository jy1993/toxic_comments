import pickle
import util
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import log_loss, roc_auc_score
import joblib

def tfidf_svm_main(col='toxic'):
	print('loading data')
	df_train = pd.read_csv(util.train_data)
	X = pickle.load(open(util.tmp_tf_idf_train, 'rb'))
	y = df_train[col]

	print('training svm model for {}...'.format(col))
	tfidf_svm = SVC(random_state=42)
	params = {
		'C': [0.1, 1],
		'kernel': ['linear', 'rbf'],
		'gamma': [0.01, 0.1]
	}
	clfs = GridSearchCV(tfidf_svm, params, scoring='neg_log_loss', cv=5, verbose=5)
	clfs.fit(X, y)

	print('best model and score:')
	best = clfs.best_estimator_
	best_score = clfs.best_score_

	print('saving model')
	joblib.dump(best, util.svm_ver0 + '_' + col + '.pkl')

if __name__ == '__main__':
	for col in util.cols:
		tfidf_svm_main(col)