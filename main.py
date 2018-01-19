import argparse
import torchtext.data as data
from model import TextCNN
import train
import torch
import random
import torch.backends.cudnn as cudnn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# data
	parser.add_argument('-data_path', type=str, default='./data', help='path to load data')
	parser.add_argument('-train', type=str, default='toxic_train.csv')
	parser.add_argument('-val', type=str, default='toxic_val.csv')
	parser.add_argument('-test', type=str, default='toxic_test.csv')

	# special for TextCNN 
	parser.add_argument('-fixed_length', type=int, default=260, help='fixed sentence length')

	# train
	parser.add_argument('-lr', type=float, default=0.001, help='set learning rate')
	parser.add_argument('-epochs', type=int, default=5, help='set epoch num')
	parser.add_argument('-batch_size', type=int, default=64, help='batch size for training')
	parser.add_argument('-patience', type=int, default=2, help='early stopping')
	parser.add_argument('-print_every', type=int, default=100, help='print every')
	parser.add_argument('-save_every', type=int, default=300, help='save every')
	parser.add_argument('-model_path', type=str, default='./model', help='where to save the model')

	# model
	parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout')
	parser.add_argument('-embedding_size', type=int, default=128, help='number of embedding dimension')
	parser.add_argument('-channel_out', type=int, default=100, help='number of each kind of kernel')
	parser.add_argument('-kernel_sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')

	# seed
	parser.add_argument('-seed', type=int, default=42, help='random seed')
	args = parser.parse_args()

	# for reproduction and model variation
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)

	# cudnn.benchmark = True

	TEXT = data.Field(fix_length=args.fixed_length, lower=True, tokenize='spacy', batch_first=True)
	LABEL = data.Field(sequential=False)
	train, val, test = data.TabularDataset.splits(path=args.data_path, 
		train=config.train, validation=config.val, test=config.test, format='csv',
		fields=[('text', TEXT), ('label', LABEL)])

	train_iter, val_iter, test_iter = data.BucketIterator.splits(
		(train, val, test), batch_sizes=(args.batch_size, 256, 256),
		sort_key=lambda x: len(x.text))

	TEXT.build_vocab(train)
	LABELS.build_vocab(train)

	args.embed_num = len(TEXT.vocab)
	args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

	# training a textCNN model
	textCNN = TextCNN(args)
	textCNN.apply(weights_init)
	train.train(train_iter, val_iter, textCNN, args)

		


