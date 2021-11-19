import torch
import torch.nn as nn
from model import LSTM, MLP_ITEM
import numpy as np
import os, sys
import time
import copy

class Ensemble_LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers=1, batch_size = 512, num_ensemble=3, device="cpu", path = ",/", initial = None, bias = True):
        super(Ensemble_LSTM, self).__init__()
        self.device = device

        # Ensemble Params
        self.num_ensemble = int(num_ensemble)

        # LSTM Params
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.item_emb = nn.Embedding(self.output_size, self.input_size, padding_idx=0)

        self.batch_size = batch_size
        print("Batch_Size for large data:",self.batch_size)
        self.path = path

        # Initializing LSTM ensemble members
        for i in range(self.num_ensemble):
            LSTM_model = LSTM(self.input_size, self.output_size, self.hidden_dim, self.n_layers, self.batch_size, self.device).to(device)
            LSTM_model.LSTM.flatten_parameters()
            setattr(self, 'lstm_'+str(i), LSTM_model)

        # Initializing MLP_item members (1 MLP for 1 item)
        for i in range(self.output_size):
            MLP_item = MLP_ITEM(self.num_ensemble, self.output_size, initial = None, bias = True).to(device)
            setattr(self, 'mlp_'+str(i), MLP_item)
            if i == 0:
                print("Item - MPL aggregator: Sample\n")
                print(MLP_item)
                print("MLP param: initial = {}, bias = {}".format(initial, bias))
                print("ZERO GRAD update")

        # sys.exit()
    
    def compute_test(self, test_data, test_labels, baseline = None):
        """
        Evaluate Test result: HITS, MRR, loss
        (1) LSTM (2) MLP for each item
        """
        # Initializing summary
        test_num = len(test_data)
        MRR,HITS,loss = 0,0,0 

        criterion = nn.CrossEntropyLoss()
        # Add: Evaluation metric
        for iteration in range(int(test_num / self.batch_size) + 1):
            st_idx, ed_idx = iteration * self.batch_size, (iteration + 1) * self.batch_size
            if ed_idx > test_num:
                ed_idx = test_num

            # Running Ensembles on a batch
            test_batch = torch.stack([self.item_emb(test_data[i]) for i in range(st_idx,ed_idx)],dim=0)
            test_label = test_labels[st_idx:ed_idx]

            # Forward
            outputs = []
            for i in range(self.num_ensemble):
                lstm = getattr(self, "lstm_"+str(i))
                output, _ = lstm.forward(test_batch)
                outputs.append(output.detach().cpu())

            #(num ensemble, batch size, num items)
            outputs = torch.stack(outputs, dim = 0)
            outputs = outputs.cpu().view(self.num_ensemble, -1, self.output_size)

            # NOTE: (Baseline) Unweighted avg. across Ensembles (Calculate at first MLP epoch)
            if baseline == "TRUE":
                #(batch size, num items)
                outputs_mean = outputs.mean(dim=0) 
                current_val = np.zeros((outputs_mean.shape[0],1))

                for i in range(st_idx,ed_idx):
                    current_test_label = test_label[i-st_idx]
                    current_val[i-st_idx,0] = outputs_mean[i-st_idx,current_test_label]

                outputs_diff = outputs_mean - current_val
                ranks = np.count_nonzero(outputs_diff > 0,axis = 1)

                for i in range(st_idx,ed_idx):
                    predicted_rank = ranks[i-st_idx]+1
                    MRR += 1/predicted_rank
                    HITS += (1 if predicted_rank<=10 else 0)

            # NOTE: (LSTM + MLP item)
            else:
                item_outputs = torch.transpose(outputs, 0, 2) #(num items, num batch size, num ensemble)
                item_outputs = item_outputs.to(self.device)
                aggr_outputs = []
                for item_number in range(self.output_size):
                    mlp = getattr(self, "mlp_"+str(item_number))
                    # (num batch size, num ensemble) > (num batch size, 1)
                    item_val = mlp.forward(item_outputs[item_number])
                    aggr_outputs.append(item_val.detach().cpu())

                # (num items, batch size, 1)
                aggr_outputs = torch.stack(aggr_outputs, dim = 0) 
                # (batch size, num items)
                aggr_outputs = aggr_outputs.squeeze().T 

                loss += criterion(aggr_outputs, test_label).item()
                current_val = np.zeros((aggr_outputs.shape[0],1))

                for i in range(st_idx,ed_idx):
                    current_test_label = test_label[i-st_idx]
                    current_val[i-st_idx,0] = aggr_outputs[i-st_idx,current_test_label]

                outputs_diff = aggr_outputs - current_val
                ranks = np.count_nonzero(outputs_diff > 0,axis = 1)

                for i in range(st_idx,ed_idx):
                    predicted_rank = ranks[i-st_idx]+1
                    MRR += 1/predicted_rank
                    HITS += (1 if predicted_rank<=10 else 0)

        if baseline == "TRUE":
            print("(Baseline) Unweighted avg.: MRR = {:.4f}, HITS = {:.4f}".format(MRR/test_num, HITS/test_num))
            self.baseline_eval.append([MRR/test_num, HITS/test_num])

        else:
            print("MLP Item Aggr. Test Evaluation: MRR = {:.4f}, HITS = {:.4f}, loss = {:.4f}".format(MRR/test_num, HITS/test_num, loss/test_num))
            self.ensemble_eval.append([MRR/test_num, HITS/test_num, loss/test_num])


    def valid_loss(self, valid_data, valid_labels):
        """
        Evaluates unweighted avg. LSTMs for optimal training (not individual)
        Calculate MRR on valid dataset
        """
        # Initializing summary
        valid_num = len(valid_data)

        BASE_MRR, BASE_HITS = 0, 0
        # criterion = nn.CrossEntropyLoss()
        for iteration in range(int(valid_num / self.batch_size) + 1):
            st_idx, ed_idx = iteration * self.batch_size, (iteration + 1) * self.batch_size
            if ed_idx > valid_num:
                ed_idx = valid_num

            # Running Ensembles on a batch
            valid_batch = torch.stack([self.item_emb(valid_data[i]) for i in range(st_idx,ed_idx)],dim=0)
            valid_label = valid_labels[st_idx:ed_idx]

            # Forward
            outputs = []
            for i in range(self.num_ensemble):
                lstm = getattr(self, "lstm_"+str(i))
                output, _ = lstm.forward(valid_batch)
                outputs.append(output)
            outputs = torch.stack(outputs, dim = 0)  #(num ensemble, num batch size, num items)

            # NOTE: (Baseline) Unweighted avg. across Ensembles
            outputs_mean = outputs.detach().cpu().mean(dim=0)
            current_val = np.zeros((outputs_mean.shape[0],1))

            for i in range(st_idx,ed_idx):
                current_test_label = valid_label[i-st_idx]
                current_val[i-st_idx,0] = outputs_mean[i-st_idx,current_test_label]

            outputs_diff = outputs_mean - current_val
            ranks = np.count_nonzero(outputs_diff > 0,axis = 1)

            for i in range(st_idx,ed_idx):
                predicted_rank = ranks[i-st_idx]+1
                BASE_MRR += 1/predicted_rank
                BASE_HITS += (1 if predicted_rank<=10 else 0)

        # save best_loss and best_epoch for each LSTM
        if self.best_MRR < BASE_MRR:
            self.best_MRR = BASE_MRR
            self.best_epoch = self.epoch


    def train(self, train, test, valid = None, epochs=50):

        # for LSTM validataion
        self.best_MRR = 0
        self.best_epoch = 0

        # for saving Final Test results    
        self.ensemble_eval = list()
        self.baseline_eval = list()

        # learning rate for LSTMs and MLPs
        lstm_lr = 1e-4
        mlp_lr = 5e-3

        print("Learning Rate: LSTM = {}, MLP-Iteim = {}".format(lstm_lr, mlp_lr))

        # Extract the data and reshape it
        train_num, test_num = len(train), len(test)

        train_data, test_data = [],[]
        train_labels, test_labels = [],[]
        
        for i in range(train_num):
            train_data.append(train[i][0])
            train_labels.append(train[i][1])
        
        train_data = torch.LongTensor(train_data).to(self.device)
        train_labels = torch.LongTensor(train_labels)

        for i in range(test_num):
            test_data.append(test[i][0])
            test_labels.append(test[i][1])
        
        test_data = torch.LongTensor(test_data).to(self.device)
        test_labels = torch.LongTensor(test_labels)

        # Add Validation
        if valid != None:
            valid_num, valid_data, valid_labels = len(valid), [], []
            for i in range(valid_num):
                valid_data.append(valid[i][0])
                valid_labels.append(valid[i][1])

            valid_data = torch.LongTensor(valid_data).to(self.device)
            valid_labels = torch.LongTensor(valid_labels)

            # self.best_loss = {lstm_number:1e+6 for lstm_number in range(self.num_ensemble)}
            # self.best_MRR = {lstm_number:0 for lstm_number in range(self.num_ensemble)}
            # self.best_epoch = {lstm_number:0 for lstm_number in range(self.num_ensemble)}

            print("train # = {}\tvalid # = {}\ttest # = {}".format(train_num,valid_num,test_num))
        else:
            print("train # = {}\ttest # = {}".format(train_num,test_num))

        # Add LSTM: Individually Trained
        optimizers = dict()
        for lstm_number in range(self.num_ensemble):
            lstm = getattr(self, "lstm_"+str(lstm_number))
            lstm = lstm.to(self.device)
            optimizers[lstm_number] = torch.optim.Adam(lstm.parameters(), lr=lstm_lr, weight_decay=1e-5)
        criterions = {i:nn.CrossEntropyLoss() for i in range(self.num_ensemble)}

        patience = 5

#============================

        for epoch in range(epochs):
            self.epoch = epoch
            print("LSTMs - Epoch is: ", str(epoch))

            # Training over every model
            for lstm_number in range(self.num_ensemble):
                lstm = getattr(self, "lstm_"+str(lstm_number))

                optimizer = optimizers[lstm_number]
                criterion = criterions[lstm_number]

                for iteration in range(int((train_num)/self.batch_size)+1):
                    st_idx, ed_idx = iteration*self.batch_size, (iteration+1)*self.batch_size
                    if ed_idx>train_num:
                        ed_idx = train_num
                        
                    optimizer.zero_grad()
                    train_batch = torch.stack([self.item_emb(train_data[i]) for i in range(st_idx,ed_idx)],dim=0)
                    output, _ = lstm.forward(train_batch.detach())
                    loss = criterion(output.cpu(), train_labels[st_idx:ed_idx])
                    loss.backward()
                    optimizer.step()

            # update valid loss for each LSTM
            self.valid_loss(valid_data, valid_labels)

            if (epoch - self.best_epoch) > patience:
                print("LSTMs are converged.")
                break
            else:
                print("Best MRR of avg. Ensemble: ", self.best_MRR/valid_num)
                print("Best Epoch: ", self.best_epoch)

            # Check Baseline result on Test data
            self.compute_test(test_data, test_labels, baseline = "TRUE")

#============================

        # Add MLP * num of items parameters
        ensemble_params = list()
        for item_number in range(self.output_size):
            mlp = getattr(self, "mlp_"+str(item_number))
            mlp = mlp.to(self.device)
            ensemble_params = ensemble_params + list(mlp.parameters())

        ensemble_optimizer = torch.optim.Adam(ensemble_params, lr=mlp_lr, weight_decay=1e-5)
        ensemble_criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.mlp_epoch = epoch
            print("MLP Item - Epoch is: ", str(epoch))

            # NOTE: (STEP2) Train MLPs (LSTMs are not trained)
            train_loss = 0
            for iteration in range(int(train_num/self.batch_size)+1):
                st_idx,ed_idx = iteration*self.batch_size, (iteration+1)*self.batch_size
                if ed_idx>train_num:
                    ed_idx = train_num

                ensemble_optimizer.zero_grad()
                train_batch = torch.stack([self.item_emb(train_data[i]) for i in range(st_idx,ed_idx)],dim=0)
                train_label = train_labels[st_idx:ed_idx]
                outputs = []
                for lstm_number in range(self.num_ensemble):
                    lstm = getattr(self, "lstm_"+str(lstm_number))
                    output, _ = lstm.forward(train_batch.detach())
                    # Adding the rank information
                    batch_size, num_items = list(output.size())[0], list(output.size())[1]
                    ranked_list = [torch.Tensor(list(range(num_items)), dtype=torch.float32)]
                    ranked_tensor = torch.stack(ranked_list, dim=0).cuda()
                    output = ranked_tensor * output
                    outputs.append(output)

                outputs = torch.stack(outputs, dim = 0)  #(num ensemble, batch size, num items)
                item_outputs = torch.transpose(outputs, 0, 2) #(num items, batch size, num ensemble)

                aggr_outputs = []
                for item_number in range(self.output_size):
                    mlp = getattr(self, "mlp_"+str(item_number))
                    # (num batch size, num ensemble) > (num batch size, 1)
                    item_scalar = mlp.forward(item_outputs[item_number].detach())
                    aggr_outputs.append(item_scalar)

                # (num items, num batch size, 1)
                aggr_outputs = torch.stack(aggr_outputs)
                ensemble_loss = ensemble_criterion(aggr_outputs.squeeze().T.cpu(), train_label)
                train_loss += ensemble_loss

                ensemble_loss.backward()
                ensemble_optimizer.step()

            print("Train Loss (LSTM sum) = {:.4f}".format(train_loss/train_num))
            self.compute_test(test_data, test_labels)

        return self.ensemble_eval, self.baseline_eval