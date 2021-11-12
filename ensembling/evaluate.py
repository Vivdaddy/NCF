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
	# print("Length of test loader", len(test_loader))
	for user, item, label in test_loader:
		user = user.cuda()
		item = item.cuda()
		label = label.cuda()
		prediction = model(user, item)
		#print("Prediction is \n", prediction)
		argmax_prediction = torch.argmax(prediction, dim=1)
		correct += argmax_prediction.eq(label.view_as(argmax_prediction)).sum().item()
		# correct += (argmax_prediction == label).float().sum()
	print("Length of test dataset is ", len(test_loader.dataset))
	print("Number correct is ", correct)
	accuracy = 100* correct / len(test_loader.dataset)
	return accuracy


def uncertainty_and_accuracy(models, test_loader):
	correct = 0
	uncertainty = torch.empty((1))
	for user, item, label in test_loader:
		user = user.cuda()
		item = item.cuda()
		label = label.cuda()
		ensemble_predictions = []
		print("ensemble predictions is ", ensemble_predictions)
		for m in models:
			prediction = torch.nn.functional.softmax(m(user, item), dim=1)
			print("predictions shape is", prediction.shape)

			ensemble_predictions.append(prediction)
		ensemble_predictions = torch.stack(ensemble_predictions)
		print("ensemble_predictions shape ", ensemble_predictions.shape)
		average_predictions = torch.mean(ensemble_predictions, dim=0)
		print("average_predictions shape ", average_predictions.shape)
		argmax_prediction = torch.argmax(average_predictions, dim=1)
		print("argmax_predictions shapoe ", argmax_prediction.shape)
		correct += argmax_prediction.eq(label.view_as(argmax_prediction)).sum().item()
		average_predictions = average_predictions.repeat(len(ensemble_predictions), 1)
		print("average predicctions shape ", average_predictions.shape)
		uncertainty += torch.nn.functional.kl_div(ensemble_predictions, average_predictions,reduction='mean')
	print("Length of test dataset is ", len(test_loader.dataset))
	print("Number correct is ", correct)
	accuracy = 100* correct / len(test_loader.dataset)
	uncertainty = uncertainty / len(test_loader.dataset)

	return accuracy, uncertainty
