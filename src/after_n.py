import pandas as pd
import util
import chardet

filepath = util.output_lstm_ver0_pre + '_copy' + '.csv'
# be lazy is great!!!
# count = 0
# id_list = []
# toxic = []
# severe_toxic = []
# obscene = []
# threat = []
# insult = []
# identity_hate = []
# with open(filepath, 'r') as f:
# 	for line in f.readlines():
# 		terms = line.strip().split(',')
# 		if len(terms) != 8:
# 			print('-' * 30)
# 			print('error, length of terms is {} does not equal to 8'.format(len(terms)))
# 			print(line, terms)
# 			break
# 		id_list += terms[0]
# 		toxic += terms[2]
# 		severe_toxic += terms[3]
# 		obscene += terms[4]
# 		threat += terms[5]
# 		insult += terms[6]
# 		identity_hate += terms[7]

# throws a UnicodeDecodeError, have no idea what's going on right now, 20171229
# seems like a encoding error
# solution: open the file with notepad ++, using utf-8 to encode the file
# next time: remeber to save as a utf-8 format file, adding encoding = 'utf-8' when calling to_csv function
df = pd.read_csv(filepath)
print(df.head())
print(df.shape)
df.drop(['comment_text'], axis=1, inplace=True)
df.to_csv(util.output_sub_lstm_ver0_pre + '.csv', index=False)