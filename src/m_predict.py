import pandas as pd
import pickle
import util
from keras.models import load_model

#loading test data
df_test = pd.read_csv(util.test_data)
#df_test.to_csv(util.output_lstm_ver0_pre + '.csv')
X_test = pickle.load(open(util.tmp_padded_seq_test, 'rb'))

#loading trained model
col_2_model = {col: load_model(util.lstm_ver0 + '_' + col) for col in util.cols}

for col in util.cols:
	df_test[col] = col_2_model[col].predict_proba(X_test, batch_size=util.batch_size, verbose=1)

print(df_test.head())
df_test.to_csv(util.output_lstm_ver0_pre + '.csv', index=False)

