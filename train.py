import torch 
from torch.optim import Adam
import torch.nn.functional as F
import os


def train(train_iter, val_iter, model, args):
	model.train()
	if torch.cuda.is_available():
		model.cuda()

	optimizer = Adam(model.parameters(), args.lr)
	step = 0
	epoch_num = args.epochs + 1
	batch_num = len(train_iter)
	min_loss = 1
	patience = args.patience
	for epoch in range(1, epoch_num):
		if patience < 1:
			break
		for batch in train_iter:
			if patience < 1:
				break
			feature, target = batch
			if torch.cuda.is_available():
				feature.cuda()
				target.cuda()

			optimizer.zero_grad()
			logits = model(feature)
			loss = F.binary_cross_entropy_with_logits(logits, target)
			loss.backward()
			optimizer.step()

			step += 1
			if step % print_every == 0:
				val_loss = val(val_iter, model, args)
				print('epoch progress: {}/{}, batch progress: {}/{}, train loss:{.6f},\
				 	val loss: {:.6f}'.format(epoch, epoch_num, step, batch_num, loss, val_loss))
				if val_loss < min_loss:
					min_loss = val_loss
					patience = args.patience
				else:
					patience -= 1

			if step % save_every == 0:
				if not os.isdir(args.model_path):
					os.mkdir(args.model_path)
				else:
					path = os.path.join(args.model_path, '_steps_{}'.format(step))
					torch.save(model.state_dict(), path)


def eval(data_iter, model, args):
	model.eval()
	avg_loss = 0

	batch_num = len(data_iter)
	for batch in data_iter:
		feature, target = batch
		if torch.cuda.is_available():
			feature.cuda()
			target.cuda()
		logits = model(feature)
		loss = F.binary_cross_entropy_with_logits(logits, target)
		avg_loss += loss

	return avg_loss/batch_num






