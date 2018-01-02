import pickle
import util
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import log_loss, roc_auc_score
import joblib
import xgboost as xgb

def tfidf_xgb_main(col='toxic'):
	print('loading data')
	df_train = pd.read_csv(util.train_data)
	X = pickle.load(open(util.tmp_tf_idf_train, 'rb'))
	y = df_train[col]
	dtrain = xgb.DMatrix(X, labels=y)

	print('training xgboost model for {}...'.format(col))
	params = {
		'objective': 'binary:logistic',
		'metrics': 'logloss',
		'eta': 0.1,
		'gamma': 1,
		'max_depth': 5,
		'min_child_weight': 2,
		'subsample': 0.5,
		'colsamplebytree': 0.5,
		'silent': 1,
		'seed': 42
	}
	clf = xgb.cv(params, dtrain, num_boost_round=100, nfold=3, stratified=True, \
				 metrics=('logloss'), early_stopping_rounds=10, verbose_eval=10)

	print('saving model')


if __name__ == '__main__':
	for col in util.cols:
		tfidf_xgb_main(col)