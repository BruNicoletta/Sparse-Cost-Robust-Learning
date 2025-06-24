import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as torch_optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt 

from timeit import default_timer as timer
import warnings
# from sklearn.model_selection._split import StratifiedShuffleSplit
# from experiments.experimental_design import experimental_design

from itertools import cycle 


torch.manual_seed(42)

def get_model(inputs, labels, ex_wts=None, params=None, is_training=True):
    #Building a simple model + computing loss and logits for the autodiff methods
    model = CSNeuralNetwork(inputs.size(1))

    # Forward pass with optional parameters
    logits = model(inputs, params=params)

    logits = logits.squeeze()
    labels = labels.squeeze()

    if ex_wts is None:
        loss = F.binary_cross_entropy_with_logits(logits, labels) 
    else:
        loss = (F.binary_cross_entropy_with_logits(logits, labels, reduction='none') * ex_wts).sum()

    return model, loss, logits 

def get_CostSensitive_model(inputs, labels, ex_wts=None, params=None, cost_matrix=None, is_training=True):
    #Building a simple model + computing loss and logits for the autodiff methods
    model = CSNeuralNetwork(inputs.size(1))

    # Forward pass with optional parameters
    logits = model(inputs, params=params)

    logits = logits.squeeze()
    labels = labels.squeeze()

    if cost_matrix is None:
        if ex_wts is None:
            #loss = nn.BCELoss()(logits, labels)
            #loss = F.binary_cross_entropy_with_logits(logits, labels)
            bce_loss = - (labels * torch.log(logits) + (1 - labels) * torch.log(1 - logits))
            loss = (bce_loss).mean()  
            #loss = F.binary_cross_entropy(logits, labels)
        else:
            #loss = nn.BCELoss(weight=ex_wts)(logits, labels)
            #loss = (F.binary_cross_entropy_with_logits(logits, labels, reduction='none') * ex_wts).mean()
            bce_loss = - (labels * torch.log(logits) + (1 - labels) * torch.log(1 - logits))
            loss = (ex_wts * bce_loss).mean()
            #loss = (F.binary_cross_entropy(logits, labels, reduction='none') * ex_wts).mean()

    else:
        if ex_wts is None:
            loss = (model.expected_cost(logits, labels, cost_matrix)).mean()
        else:
            loss = (model.expected_cost(logits, labels, cost_matrix) * ex_wts).mean()

    return model, loss, logits 

class CostInsensitiveDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CostSensitiveDataset(Dataset):
    def __init__(self, X, y, w):
        self.X = X
        self.y = y
        self.w = w

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]


class CSNeuralNetwork(nn.Module):
    def __init__(self, n_inputs, obj='ce', lambda1=0, lambda2=0, n_neurons=16, directory='', training_cost_matrix_available = True):
        super().__init__()

        self.n_inputs = n_inputs
        self.cost_sensitive = (obj == 'weightedce' or obj == 'aec'or obj == 'reweighting_method') 
        self.obj = obj
        self.training_cost_matrix_available = training_cost_matrix_available

        self.lin_layer1 = nn.Linear(n_inputs, n_neurons)
        self.final_layer = nn.Linear(n_neurons, 1)
        self.sigmoid = nn.Sigmoid()

        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.directory = directory


    def forward(self, x, params=None):
        if params is not None:
            w1, b1, w2, b2 = params
            x = torch.tanh(F.linear(x, w1, b1))
            x = F.linear(x, w2, b2)
        else:
            x = self.lin_layer1(x)
            x = torch.tanh(x)
            x = self.final_layer(x)
        x = self.sigmoid(x) # Could probably be removed and use  F.binary_cross_entropy_with_logits instead of
        return x
    

    def classic_reweight_autodiff(self, inp_a, label_a, inp_b, label_b, stepsize, eps=0.0):
        # Initialize example weights for model_a
        #ex_wts_a = torch.ones(inp_a.size(0), requires_grad=True, dtype=torch.float32) / float(inp_a.size(0))
        ex_wts_a = torch.zeros(inp_a.size(0), requires_grad=True, dtype=torch.float32) 
        ex_wts_b = torch.ones(inp_b.size(0), dtype=torch.float32) / float(inp_b.size(0))
        
        # Get model for noisy pass (inp_a, label_a)
        model_a, loss_a, logits_a = get_model(inp_a, label_a, ex_wts=ex_wts_a, is_training=True) 
        
        # Compute gradients of the loss w.r.t model_a's parameters
        grads_a = torch.autograd.grad(loss_a, model_a.parameters(), create_graph=True)
        
        # Update parameters explicitly incorporating `ex_wts_a`
        new_params_a = [param - stepsize * grad for param, grad in zip(model_a.parameters(), grads_a)]
        
        # Create a model_b using the updated parameters
        # Compute the loss for model_b
        model_b, loss_b, logits_b = get_model(inp_b, label_b, ex_wts=ex_wts_b, params=new_params_a, is_training=True)
        
        # Compute gradients of loss_b w.r.t `ex_wts_a`
        #grads_ex_wts_cheating = torch.autograd.grad(loss_b, ex_wts_a, allow_unused=True)[0]
        grads_ex_wts = torch.autograd.grad(loss_b, ex_wts_a)[0]

        # --- Compute new example/ noisy sample weights based on the grads_ex_wts  ---
        ex_weight = -grads_ex_wts
        ex_weight_plus = torch.clamp(ex_weight, min=eps)
        ex_weight_sum = ex_weight_plus.sum()
        ex_weight_sum += torch.eq(ex_weight_sum, 0.0).type(ex_weight_sum.dtype) 
        """
        "ex_weight_sum += torch.eq(ex_weight_sum, 0.0).type(ex_weight_sum.dtype)"
        Can probably be written by this instead as technically ex_weight_sum is a scalar, not a tensor. 
        However I kept it as a "tensor" following the logic (and tensorFlow implementation) from Mengye Ren et Al. : 
        # Avoid division by zero
        if ex_weight_sum == 0.0:
        ex_weight_sum = 1.0
        """ 
        ex_weight_norm = ex_weight_plus / ex_weight_sum 

        return ex_weight_norm 
    
    def sparse_costs_reweight_autodiff(self, inp_a, label_a, inp_b, label_b, stepsize, eps=0.0, noisy_cost_matrix=None, clean_cost_matrix=None):
        # Initialize example weights for model_a
        #ex_wts_a = torch.ones(inp_a.size(0), requires_grad=True, dtype=torch.float32) / float(inp_a.size(0))
        ex_wts_a = torch.zeros(inp_a.size(0), requires_grad=True, dtype=torch.float32) 
        ex_wts_b = torch.ones(inp_b.size(0), dtype=torch.float32) / float(inp_b.size(0))
        
        # Get model for noisy pass (inp_a, label_a)
        #model_a, loss_a, logits_a = get_CostSensitive_model(inp_a, label_a, ex_wts=ex_wts_a, cost_matrix=noisy_cost_matrix, is_training=True)
        params_a = [torch.nn.Parameter(p.clone()) for p in self.parameters()] ###### To be deleted /changed
        params_a_rework = [p.clone().detach().requires_grad_(True) for p in self.parameters()]
        model_a, loss_a, logits_a = get_CostSensitive_model(inp_a, label_a, ex_wts=ex_wts_a, params= params_a_rework, cost_matrix=noisy_cost_matrix, is_training=True) ###### To be deleted /changed
        
        """
        print('\n\n***********************\n')
        print('sparse_costs_reweight_autodiff \n\n\n')  
        print(f'params_a Shape: {len(params_a)} \n \n')
        print(f'params_a: {params_a}\n \n')   
        print(f'params_a_rework Shape: {len(params_a_rework)}\n \n')
        print(f'params_a_rework: {params_a_rework}\n \n')  
        """
             
        
        # Compute gradients of the loss w.r.t model_a's parameters
        grads_a = torch.autograd.grad(loss_a, params_a_rework, create_graph=True) ###### To be deleted /changed
        #grads_a = torch.autograd.grad(loss_a, model_a.parameters(), create_graph=True)
        
        """  
        print('\n\n***********************\n')
        print('sparse_costs_reweight_autodiff \n\n\n')  
        print(f'grads_a Shape: {len(grads_a)}\n \n')
        print(f'grads_a: {grads_a}\n \n')
        """
        
        # Update parameters explicitly incorporating `ex_wts_a`
        #new_params_a = [param - stepsize * grad for param, grad in zip(model_a.parameters(), grads_a)]
        new_params_a = [param - stepsize * grad for param, grad in zip(params_a_rework, grads_a)]

        
        # Create a model_b using the updated parameters
        # Compute the loss for model_b
        model_b, loss_b, logits_b = get_CostSensitive_model(inp_b, label_b, ex_wts=ex_wts_b, params=new_params_a, cost_matrix=clean_cost_matrix, is_training=True) 
        
        # Compute gradients of loss_b w.r.t `ex_wts_a`
        #grads_ex_wts_cheating = torch.autograd.grad(loss_b, ex_wts_a, allow_unused=True)[0]
        grads_ex_wts = torch.autograd.grad(loss_b, ex_wts_a)[0]
        """ 
        print('\n\n***********************\n')
        print('sparse_costs_reweight_autodiff \n\n\n')  
        print(f'grads_ex_wts Shape: {len(grads_ex_wts)}\n \n')
        print(f'grads_ex_wts: {grads_ex_wts}\n \n')
        """

        # --- Compute new example/ noisy sample weights based on the grads_ex_wts  ---
        ex_weight = -grads_ex_wts
        ex_weight_plus = torch.clamp(ex_weight, min=eps)
        ex_weight_sum = ex_weight_plus.sum()
        ex_weight_sum += torch.eq(ex_weight_sum, 0.0).type(ex_weight_sum.dtype) 
        """
        "ex_weight_sum += torch.eq(ex_weight_sum, 0.0).type(ex_weight_sum.dtype)"
        Can probably be written by this instead as technically ex_weight_sum is a scalar, not a tensor. 
        However I kept it as a "tensor" following the logic (and tensorFlow implementation) from Mengye Ren et Al. : 
        # Avoid division by zero
        if ex_weight_sum == 0.0:
        ex_weight_sum = 1.0
        """
        ex_weight_norm = ex_weight_plus / ex_weight_sum 

        return ex_weight_norm 


    def model_train(self, model, x_train, y_train, x_val, y_val, cost_matrix_train=None, cost_matrix_val=None,
                    n_epochs=500, verbose=True):

        losses = [] 
        last_epoch = 0

        """
        print("cost_matrix_train:", cost_matrix_train.shape)
        print("cost_matrix_val:", cost_matrix_val.shape)
        print("x_train:", x_train.shape)
        print("y_train:", y_train.shape)
        print("x_val:", x_val.shape)
        print("y_val:", y_val.shape)
        """
        
        # Settings:
        batch_size = max(8, int(len(x_train) / 100))  # Originally 2 ** 10
        batch_size = min(batch_size, 256)

        """ 
        print('\n\n***********************\n')
        print('Initial setting \n\n\n')  
        print(f'x_train: {x_train.shape}\n \n')
        print(f'y_train: {y_train.shape}\n \n')   
        print(f'batch_size: {batch_size}\n \n')     
        """
        early_stopping_criterion = 25

        # Move to GPU:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        if self.cost_sensitive:
            print('self.training_cost_matrix_available:')
            if self.training_cost_matrix_available:
                train_ds = CostSensitiveDataset(torch.from_numpy(x_train).float(),
                                                torch.from_numpy(y_train[:, None]).float(),
                                                torch.from_numpy(cost_matrix_train))
                val_ds = CostSensitiveDataset(torch.from_numpy(x_val).float(),
                                            torch.from_numpy(y_val[:, None]).float(),
                                            torch.from_numpy(cost_matrix_val))
            else:
                train_ds = CostInsensitiveDataset(torch.from_numpy(x_train).float(),
                                                torch.from_numpy(y_train[:, None]).float())
                val_ds = CostSensitiveDataset(torch.from_numpy(x_val).float(),
                                            torch.from_numpy(y_val[:, None]).float(),
                                            torch.from_numpy(cost_matrix_val))        
        else:
            train_ds = CostInsensitiveDataset(torch.from_numpy(x_train).float(),
                                              torch.from_numpy(y_train[:, None]).float())
            val_ds = CostInsensitiveDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val[:, None]).float())

        optimizer = torch_optim.Adam(model.parameters(), lr=0.001)

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=int(batch_size / 4), shuffle=True)

        best_val_loss = float("Inf")

        epochs_not_improved = 0

        for epoch in range(n_epochs):
            start = timer()

            running_loss = 0.0

            # Training
            model.train() 

            # Combine the iterators for training and validation data
            val_iter = cycle(val_dl)

            for i, data in enumerate(train_dl):
                if self.cost_sensitive and self.training_cost_matrix_available:
                    inputs, labels, cost_matrix_batch = data
                    inputs, labels, cost_matrix_batch = inputs.to(device), labels.to(device), cost_matrix_batch.to(
                        device)
                else:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    cost_matrix_batch = None 


                ######################################
                ##  Added for reweight and tryouts

                # Get a batch of validation data
                val_data = next(val_iter)
                if self.cost_sensitive:
                    val_inputs, val_labels, val_cost_matrix = val_data
                    val_inputs, val_labels, val_cost_matrix = val_inputs.to(device), val_labels.to(
                                                                device), val_cost_matrix.to(device)
                else:    
                    val_inputs, val_labels = val_data
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    val_cost_matrix = None 

                # Rajouter une boucle pour chaque sous-batch ; en gros il faut séparer entre 
                # le Noisy batch pour le noisy pass (feeding le modèle A)
                # le Clean batch for the clean pass (feeding le modèle B) 

                """ 
                print('\n\n***********************\n')
                print('Info Loop\n') 
                print(f'iter: {i}\n \n') 
                print(f'iter data: {data}\n \n')
                print('\n\n***********************\n')
                print('After DataLoader \n\n\n')  
                print(f'inputs: {inputs.shape}\n \n')
                print(f'labels: {labels.shape}\n \n')
                print(f'cost_matrix_batch: {cost_matrix_batch.shape}\n \n')
                print(f'val_inputs: {labels.shape}\n \n')
                print(f'val_labels: {labels.shape}\n \n')
                print(f'val_cost_matrix: {labels.shape}\n \n')
                """

                """
                ==> VERY IMPORTANT but probably we can just adapt the batch we create for the runs
                """

                """
                print('\n\n***********************\n')
                print('Before classic_reweight_autodiff \n\n\n')        
                print(f'batch_size: {batch_size}\n \n')
                """

                #  ------------------------------------------------------------------ |
                #  If someone whishes to try the classical "reweight_autodiff"   
                """ 
                CS_sample_weights = model.classic_reweight_autodiff(
                        inputs, labels, val_inputs, val_labels, 
                        stepsize=0.1, 
                        eps=0.0
                    )
                """
                #  ------------------------------------------------------------------ |

                # Introduire: les couts 
   
                if self.obj == 'reweighting_method':
                    CS_sample_weights = model.sparse_costs_reweight_autodiff(
                        inputs, labels, val_inputs, val_labels, 
                        stepsize=1, 
                        eps=0.0,
                        noisy_cost_matrix= cost_matrix_batch, 
                        clean_cost_matrix= val_cost_matrix 
                    )
  



                """ 
                # Actually not necessary 
                if self.obj == 'reweighting_method':
                    if self.training_cost_matrix_available == True: 
                        CS_sample_weights = model.sparse_costs_reweight_autodiff(
                            inputs, labels, val_inputs, val_labels, 
                            stepsize=0.1, 
                            eps=0.0,
                            noisy_cost_matrix= cost_matrix_train, 
                            clean_cost_matrix= cost_matrix_val 
                        )
                    else: 
                        CS_sample_weights = model.sparse_costs_reweight_autodiff(
                            inputs, labels, val_inputs, val_labels, 
                            stepsize=0.1, 
                            eps=0.0,
                            noisy_cost_matrix= None, 
                            clean_cost_matrix= cost_matrix_val
                        )    
                """


                #print('\n\n***********************\n')
                #print('At the end of classic_reweight_autodiff \n\n\n')        
                #print(f'CS_sample_weights: {CS_sample_weights}\n \n')

                """
                Very important to add here the !! 
                """
                #CS_sample_weights = model.autodiff_CS_reweighting(inputs , labels, val_inputs, val_labels, bsize_a = batch_size, bsize_b = int(batch_size / 4), eps=0.0)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)

                if self.obj == 'ce':
                    criterion = nn.BCELoss()
                    loss = criterion(outputs, labels)
                elif self.obj == 'weightedce':
                    misclass_cost_batch = torch.zeros((len(labels), 1), dtype=torch.double, device=device)
                    misclass_cost_batch[labels == 0] = cost_matrix_batch[:, 1, 0][:, None][labels == 0]
                    misclass_cost_batch[labels == 1] = cost_matrix_batch[:, 0, 1][:, None][labels == 1]

                    loss = nn.BCELoss(weight=misclass_cost_batch)(outputs, labels)
                elif self.obj == 'aec':
                    loss = (self.expected_cost(outputs, labels, cost_matrix_batch)).mean()
                elif self.obj == 'reweighting_method':
                    #loss = (self.expected_cost(outputs, labels, cost_matrix_batch)*CS_sample_weights).mean() 
                    #loss = (F.binary_cross_entropy_with_logits(outputs, labels, reduction='none') * CS_sample_weights).mean()
                    #loss = nn.BCELoss(weight=CS_sample_weights)(outputs, labels)
                    bce_loss = - (labels * torch.log(outputs) + (1 - labels) * torch.log(1 - outputs))
                    loss = (CS_sample_weights * bce_loss).mean()
                    #loss = (F.binary_cross_entropy(outputs, labels, reduction='none') * CS_sample_weights).mean()


                else:
                    print(self.obj)
                    raise Exception('Objective function not recognized')

                # Add regularization
                model_params = torch.cat([params.view(-1) for params in model.parameters()])
                l1_regularization = self.lambda1 * torch.norm(model_params, 1)
                # print('l1 regularization = %.5f' % l1_regularization)
                l2_regularization = self.lambda2 * torch.norm(model_params, 2)**2  # torch.norm returns the square root
                # print('l2 regularization = %.5f' % l2_regularization)
                loss += l1_regularization + l2_regularization

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            # last_epoch = epoch+1
            # losses.append(loss.detach().numpy()) 
            mid_time = timer()

            # Validation check
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for val_i, val_data in enumerate(val_dl):
                    if self.obj == 'ce':
                        val_inputs, val_labels = val_data
                        val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                        val_outputs = model(val_inputs)
                        val_loss = criterion(val_outputs, val_labels)

                    elif self.obj == 'weightedce':
                        val_inputs, val_labels, val_cost_matrix = val_data
                        val_inputs, val_labels, val_cost_matrix = val_inputs.to(device), val_labels.to(
                                                                    device), val_cost_matrix.to(device)
                        val_outputs = model(val_inputs)

                        misclass_cost_val = torch.zeros((len(val_labels), 1), dtype=torch.double, device=device)
                        misclass_cost_val[val_labels == 0] = val_cost_matrix[:, 1, 0][:, None][val_labels == 0]
                        misclass_cost_val[val_labels == 1] = val_cost_matrix[:, 0, 1][:, None][val_labels == 1]

                        val_loss = nn.BCELoss(weight=misclass_cost_val)(val_outputs, val_labels)

                    elif self.obj == 'aec' or self.obj == 'reweighting_method':
                        val_inputs, val_labels, val_cost_matrix = val_data
                        val_inputs, val_labels, val_cost_matrix = val_inputs.to(device), val_labels.to(
                                                                    device), val_cost_matrix.to(device)
                        val_outputs = model(val_inputs)
                        val_loss = (self.expected_cost(val_outputs, val_labels, val_cost_matrix)).mean()
                    
                    total_val_loss += val_loss

            end = timer()

            if total_val_loss < best_val_loss:

                # Is improvement large enough?
                # Not if difference in val_loss is < 10**-2
                if best_val_loss - total_val_loss < 10**-2:
                    epochs_not_improved += 1
                    # print('\t\tDifference: {}'.format(best_val_loss - total_val_loss))
                    if epochs_not_improved > early_stopping_criterion:
                        print(
                            '\t\tEarly stopping criterion reached: validation loss not significantly improved for {}'
                            ' epochs.'.format(
                                epochs_not_improved - 1))
                        print('\t\tInsufficient improvement in validation loss')
                        break
                else:
                    epochs_not_improved = 0

                best_val_loss = total_val_loss

                checkpoint = {
                    'epoch': epoch + 1,
                    'best validation loss': best_val_loss,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, self.directory + 'checkpoint')

                if verbose:
                    if epoch % 1 == 0:
                        print('\t\t[Epoch %d]\tloss: %.8f\tval_loss: %.8f\tTime [s]: %.2f (%.2f)\tModel saved!' % (
                        epoch + 1, running_loss / len(train_ds), total_val_loss / len(val_ds) / 4, end - start,
                        mid_time - start))

            else:
                epochs_not_improved += 1
                if epochs_not_improved > early_stopping_criterion:
                    print('\t\tEarly stopping criterion reached: validation loss not significantly improved for {}'
                          ' epochs.'.format(
                        epochs_not_improved - 1))
                    break

                if verbose:
                    if epoch % 10 == 9:
                        print('\t\t[Epoch %d]\tloss: %.8f\tval_loss: %.8f\tTime [s]: %.2f (%.2f)\tModel saved!' % (
                        epoch + 1, running_loss / len(train_ds), total_val_loss / len(val_ds) / 4, end - start,
                        mid_time - start))

        # Load last saved checkpoint:
        best_checkpoint = torch.load(self.directory + 'checkpoint')
        model.load_state_dict(best_checkpoint['model'])

        if verbose:
            print('\tFinished training! Best validation loss at epoch %d (loss: %.8f)\n'
                  % (best_checkpoint['epoch'], best_val_loss / len(val_ds) / 4))

        if best_checkpoint['epoch'] > (n_epochs - early_stopping_criterion):
            warnings.warn('Number of epochs might have to be increased!')

        """ 
        # # Graph it out! 
        plt.plot(range(last_epoch), losses)
        plt.ylabel("loss/error")
        plt.xlabel("Number of Epochs")
        plt.show()  
        """
        
        return model.to('cpu')

    def model_predict(self, model, X_test):
        y_pred = torch.zeros(len(X_test)).float()

        test_ds = CostInsensitiveDataset(torch.from_numpy(X_test).float(), y_pred)  # Amounts only needed for loss

        test_dl = DataLoader(test_ds, batch_size=X_test.shape[0])

        preds = []

        model.eval()

        with torch.no_grad():
            for x, _ in test_dl:
                prob = model(x)
                preds.append(prob.flatten())

        return preds[0].numpy()

    """ 
    #The old one with the mean  (as reminder if issues / need to debug old code)
    def expected_cost(self, output, target, cost_matrix):

        ec = target * (output * cost_matrix[:, 1, 1] + (1 - output) * cost_matrix[:, 0, 1]) \
              + (1 - target) * (output * cost_matrix[:, 1, 0] + (1 - output) * cost_matrix[:, 0, 0])

        return ec.mean()
    """

       
    def expected_cost(self, output, target, cost_matrix):

        ec = target * (output * cost_matrix[:, 1, 1] + (1 - output) * cost_matrix[:, 0, 1]) \
              + (1 - target) * (output * cost_matrix[:, 1, 0] + (1 - output) * cost_matrix[:, 0, 0])

        return ec
    

    """ 
    def expected_cost(self, output, target, cost_matrix):
        # Ensure cost_matrix is a PyTorch tensor
        if isinstance(cost_matrix, np.ndarray):
            cost_matrix = torch.tensor(cost_matrix, dtype=output.dtype, device=output.device)

        # If cost_matrix is a 2x2 matrix, expand it to match batch size
        if cost_matrix.dim() == 2:
            cost_matrix = cost_matrix.unsqueeze(0).repeat(output.size(0), 1, 1)

        # Reshape output and target for broadcasting
        output = output.unsqueeze(-1)  # (batch_size,) -> (batch_size, 1)
        target = target.unsqueeze(-1)  # (batch_size,) -> (batch_size, 1)

        # Compute the expected cost
        ec = target * (output * cost_matrix[:, 1, 1] + (1 - output) * cost_matrix[:, 0, 1]) \
            + (1 - target) * (output * cost_matrix[:, 1, 0] + (1 - output) * cost_matrix[:, 0, 0])

        return ec
    """

    def tune(self, l1, lambda1_list, l2, lambda2_list, neurons_list, x_train, y_train, cost_matrix_train, x_val, y_val,
             cost_matrix_val):

        results = np.ones((3, len(neurons_list)))
        results[0, :] = neurons_list

        for i, n_neurons in enumerate(neurons_list):
            print('Number of neurons: {}'.format(n_neurons))

            if l1:
                self.lambda2 = 0
                losses_list_l1 = []
                for lambda1 in lambda1_list:
                    net = CSNeuralNetwork(n_inputs=x_train.shape[1], obj=self.obj, lambda1=lambda1, n_neurons=n_neurons,
                                          directory=self.directory, training_cost_matrix_available = self.training_cost_matrix_available)

                    net = net.model_train(net, x_train, y_train, x_val, y_val,
                                          cost_matrix_train=cost_matrix_train, cost_matrix_val=cost_matrix_val)

                    scores_val = net.model_predict(net, x_val)

                    # Evaluate loss (without regularization term!)
                    net.lambda1 = 0
                    if self.obj == 'ce':
                        eps = 1e-9  # small value to avoid log(0)
                        ce = - (y_val * np.log(scores_val + eps) + (1 - y_val) * np.log(1 - scores_val + eps))
                        val_loss = ce.mean()
                    elif self.obj == 'weightedce':
                        eps = 1e-9  # small value to avoid log(0)
                        ce = - (y_val * np.log(scores_val + eps) + (1 - y_val) * np.log(1 - scores_val + eps))

                        cost_misclass = np.zeros(len(y_val))
                        cost_misclass[y_val == 0] = cost_matrix_val[:, 1, 0][y_val == 0]
                        cost_misclass[y_val == 1] = cost_matrix_val[:, 0, 1][y_val == 1]

                        weighted_ce = cost_misclass * ce
                        val_loss = weighted_ce.mean()
                    elif self.obj == 'aec' or self.obj == 'reweighting_method':
                        def aec_val(scores, y_true):
                            ec = y_true * (
                                 scores * cost_matrix_val[:, 1, 1] + (1 - scores) * cost_matrix_val[:, 0, 1])\
                                 + (1 - y_true) * (
                                 scores * cost_matrix_val[:, 1, 0] + (1 - scores) * cost_matrix_val[:, 0, 0])

                            return ec.mean()

                        aec = aec_val(scores_val, y_val)
                        val_loss = aec

                    print('\t\tLambda l1 = %.5f;\tLoss = %.5f' % (lambda1, val_loss))
                    losses_list_l1.append(val_loss)
                lambda1_opt = lambda1_list[np.argmin(losses_list_l1)]
                print('\tOptimal lambda = %.5f' % lambda1_opt)
                self.lambda1 = lambda1_opt

                results[1, i] = lambda1_opt
                results[2, i] = np.min(losses_list_l1)

            elif l2:
                self.lambda1 = 0
                losses_list_l2 = []
                for lambda2 in lambda2_list:
                    net = CSNeuralNetwork(n_inputs=x_train.shape[1], obj=self.obj, lambda2=lambda2, n_neurons=n_neurons,
                                          directory=self.directory)

                    net = net.model_train(net, x_train, y_train, x_val, y_val,
                                          cost_matrix_train=cost_matrix_train, cost_matrix_val=cost_matrix_val)

                    scores_val = net.model_predict(net, x_val)

                    # Evaluate loss (without regularization term!)
                    net.lambda2 = 0
                    if self.obj == 'ce':
                        eps = 1e-9
                        ce = - (y_val * np.log(scores_val + eps) + (1 - y_val) * np.log(1 - scores_val + eps))
                        val_loss = ce.mean()
                    elif self.obj == 'weightedce':
                        eps = 1e-9
                        ce = - (y_val * np.log(scores_val + eps) + (1 - y_val) * np.log(1 - scores_val + eps))

                        cost_misclass = np.zeros(len(y_val))
                        cost_misclass[y_val == 0] = cost_matrix_val[:, 1, 0][y_val == 0]
                        cost_misclass[y_val == 1] = cost_matrix_val[:, 0, 1][y_val == 1]

                        weighted_ce = cost_misclass * ce
                        val_loss = weighted_ce.mean()
                    elif self.obj == 'aec' or self.obj == 'reweighting_method':
                        def aec_val(scores, y_true):
                            ec = y_true * (
                                 scores * cost_matrix_val[:, 1, 1] + (1 - scores) * cost_matrix_val[:, 0, 1])\
                                 + (1 - y_true) * (
                                 scores * cost_matrix_val[:, 1, 0] + (1 - scores) * cost_matrix_val[:, 0, 0])

                            return ec.mean()

                        aec = aec_val(scores_val, y_val)
                        val_loss = aec


                    print('\t\tLambda l2 = %.5f;\tLoss = %.5f' % (lambda2, val_loss))
                    losses_list_l2.append(val_loss)
                lambda2_opt = lambda2_list[np.argmin(losses_list_l2)]
                print('\tOptimal lambda = %.5f' % lambda2_opt)
                self.lambda2 = lambda2_opt

                results[1, i] = lambda2_opt
                results[2, i] = np.min(losses_list_l2)

            else:
                self.lambda1 = 0
                self.lambda2 = 0
                net = CSNeuralNetwork(n_inputs=x_train.shape[1], obj=self.obj, n_neurons=n_neurons,
                                      directory=self.directory, training_cost_matrix_available = self.training_cost_matrix_available)

                net = net.model_train(net, x_train, y_train, x_val, y_val, cost_matrix_train=cost_matrix_train,
                                      cost_matrix_val=cost_matrix_val, verbose=True)

                scores_val = net.model_predict(net, x_val)

                if self.obj == 'ce':
                    eps = 1e-9
                    ce = - (y_val * np.log(scores_val + eps) + (1 - y_val) * np.log(1 - scores_val + eps))
                    val_loss = ce.mean()
                elif self.obj == 'weightedce':
                    eps = 1e-9
                    ce = - (y_val * np.log(scores_val + eps) + (1 - y_val) * np.log(1 - scores_val + eps))

                    cost_misclass = np.zeros(len(y_val))
                    cost_misclass[y_val == 0] = cost_matrix_val[:, 1, 0][y_val == 0]
                    cost_misclass[y_val == 1] = cost_matrix_val[:, 0, 1][y_val == 1]

                    weighted_ce = cost_misclass * ce
                    val_loss = weighted_ce.mean()
                elif self.obj == 'aec' or self.obj == 'reweighting_method':
                    def aec_val(scores, y_true):
                        ec = y_true * (
                             scores * cost_matrix_val[:, 1, 1] + (1 - scores) * cost_matrix_val[:, 0, 1]) \
                             + (1 - y_true) * (
                             scores * cost_matrix_val[:, 1, 0] + (1 - scores) * cost_matrix_val[:, 0, 0])

                        return ec.mean()

                    aec = aec_val(scores_val, y_val)
                    val_loss = aec
                print('\t\tNumber of neurons = %i;\tLoss = %.5f' % (n_neurons, val_loss))
                results[2, i] = val_loss

        # Assign best settings
        opt_ind = np.argmin(results[2, :])
        opt_n_neurons = int(results[0, opt_ind])
        print('Optimal number of neurons: {}'.format(opt_n_neurons))
        if l1:
            self.lambda1 = results[1, opt_ind]
            print('Optimal l1: {}'.format(self.lambda1))
        if l2:
            self.lambda2 = results[1, opt_ind]
            print('Optimal l2: {}'.format(self.lambda2))

        return CSNeuralNetwork(self.n_inputs, self.obj, self.lambda1, self.lambda2, opt_n_neurons, self.directory, self.training_cost_matrix_available)

    