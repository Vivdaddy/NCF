import numpy as np
import torch


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k):
	HR, NDCG = [], []

	for user, item, label in test_loader:
		user = user.cuda()
		item = item.cuda()

		predictions = model(user, item)
		print("predictions \n", predictions)
		print("topk ", top_k)
		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(
				item, indices).cpu().numpy().tolist()

		gt_item = item[0].item()
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))

	return np.mean(HR), np.mean(NDCG)


def accuracy(model, test_loader):
	correct = 0
	for user, item, label in test_loader:
		user = user.cuda()
		item = item.cuda()
		label = label.cuda()
		prediction = model(user, item)
		argmax_prediction = torch.argmax(prediction, dim=1)
		correct += argmax_prediction.eq(label.view_as(argmax_prediction)).sum().item()
		# correct += (argmax_prediction == label).float().sum()
	print("Length of test dataset is ", len(test_loader.dataset))
	accuracy = 100* correct / len(test_loader.dataset)
	return accuracy
