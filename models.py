import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
	def __init__(self, args):
		super(TextCNN, self).__init__()
		self.args = args
		
		V = args.vocab_size
		D = args.embedding_size
		C_in = 1
		C_out = args.channel_out
		ks = args.kernel_sizes
		droput = args.droput
		self.embedding = nn.Embedding(V, D)
		self.convs = nn.ModuleList([nn.Conv2d(C_in, C_out, (k, D)) for k in ks])
		self.droput = nn.Dropout(dropuut)
		self.fc = nn.Linear(len(ks) * C_out, 1)
		self.init_weight()

	def forward(self, x):
		emb = self.embedding(x)  # N, H, W
		x = x.unsqueeze(1)   # N, 1, H, W
		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # N, C, H
		x = [nn.MaxPool1d(i, i.size(2)).unsqueeze(2) for i in x] #N, C
		x = torch.cat(x, 1) 
		x = self.droput(x)
		logit = self.fc(x)
		return logit

	def init_weight(self):
		self.fc.weight.data.randn_()
		self.fc.bias.data.fill_(0)

class TextLSTM(nn.Module):
	def __init__(self, args):
		super(TextLSTM, self).__init__()
		V = args.vocab_size
		D = args.embedding_size
		num_layers = args.num_layers

	def forward(self, x):
		pass

