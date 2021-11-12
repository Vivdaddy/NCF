# dataset name 
dataset = 'lastfm'
assert dataset in ['ml-1m', 'pinterest-20', 'lastfm']

# model name 
model = 'NeuMF-end'
assert model in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']

# paths
main_path = '/home/vanand37/Research/NCF/ensembling/data/'

# train_rating = main_path + '{}.train.rating'.format(dataset)
# test_rating = main_path + '{}.test.rating'.format(dataset)
# test_negative = main_path + '{}.test.negative'.format(dataset)

train_rating = main_path + '{}.tsv'.format(dataset)

model_path = './models/'
GMF_model_path = model_path + 'GMF.pth'
MLP_model_path = model_path + 'MLP.pth'
NeuMF_model_path = model_path + 'NeuMF.pth'
