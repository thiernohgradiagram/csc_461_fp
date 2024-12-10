import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler
import sys
import os
from sklearn.preprocessing import StandardScaler

#Richard's Neural Network

class MLP(nn.Module):
  def __init__(self, in_dim, out_dim, hid_layers):
    super(MLP, self).__init__()
    self.nn_layers = [nn.Flatten()] #this will hold all the layers of the neural netowrk. Flatten is to make sure that the shapes are all the same to avoid any mismatches. 
    #input_dim = in_dim
    for i in range (len(hid_layers)):
      hidden_dim = hid_layers[i]
      if i == 0: #first input layer. This takes the specified input dimension and the dimension of first hidden layer
        self.nn_layers.append(nn.Linear(in_dim, hidden_dim))
        
      else: #the hidden layers.
        self.nn_layers.append(nn.Linear(input_dim, hidden_dim))
        self.nn_layers.append(nn.BatchNorm1d(hidden_dim))
      #this will connect the layers together by storing the dimension of the output from the layer for the next layer.
      input_dim = hidden_dim
      #apply the relu activation function
      self.nn_layers.append(nn.ReLU())
      
    #output layer  
    self.nn_layers.append(nn.Linear(input_dim, out_dim))

    self.net = nn.Sequential(*self.nn_layers)


  def forward(self, x):
    #flatten the input
    return self.net(x)
  
  def test_model(self, model, criterion, loader, device):
    chosen_model = model
    
    #store the values of each batch in these variables
    correct_predictions = 0
    total_loss = 0.0

    #set the model to evaluation mode
    chosen_model.eval()
    #disable gradient computation
    with torch.no_grad():
        for i, batch in enumerate(loader):
            #first get the batch data
            x = batch[0].to(device) # input data(features)
            y = batch[1].to(device) # target labels that corresponds with the input data
            logits = chosen_model(x)
            #calculate the loss. The outpus and the labels
            batch_loss = criterion(logits,y)
            
            total_loss += (batch_loss.item() * x.size(0))

            #get the predictions so the accuracy can be calculated
            preds = torch.argmax(logits, dim=1)
            #calculate the accuracy of the current batch. .sum counts the number of correct predictions for the tensor (preds == y). That looks at all the predictions in the batch against the correct label. 
            correct_predictions += ((preds == y).sum().item() / x.size(0))


    #calculate the average loss
    average_loss = total_loss / len(loader.dataset)
    #calculate the average accuracy accross all the batches by comparing the number of correct predictions to the total number of samples.        
    average_accuracy = correct_predictions / len(loader)

    return average_loss, average_accuracy
  
  #this will train the model on the data
  def train_model(self, model, criterion, optimizer, tr_loader, va_loader, n_epochs, device):
    chosen_model = model
    
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    #go through each epoch
    for epoch in range(n_epochs):
        #set the model to train mode
        chosen_model.train()

        #for training data
        training_epoch_loss = 0
        training_epoch_accuracy = 0

        #for validation data
        validation_epoch_loss = 0
        validation_epoch_accuracy = 0
        

        #go through all the training minibatches
        for i, batch in enumerate(tr_loader): 
            #retrieve the data
            x = batch[0].to(device)
            y = batch[1].to(device)
            #perform the forward pass
            logits = chosen_model(x)
            #compute the loss of this batch
            loss = criterion(logits, y)
            training_epoch_loss += (loss.item()* x.size(0))
            #compute the accuracy of the current batch
            preds = torch.argmax(logits, dim=1)
            training_epoch_accuracy += ((preds == y).sum() / x.size(0))
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #calculate the accuracy and loss on the training data
        current_training_epoch_loss = training_epoch_loss/len(tr_loader.dataset)
        training_loss.append(current_training_epoch_loss)
        current_epoch_accuracy = training_epoch_accuracy/len(tr_loader)
        training_accuracy.append(current_epoch_accuracy)

        chosen_model.eval()
        with torch.no_grad():
          #go through the batches of validation data
          for i,batch in enumerate(va_loader):
              x = batch[0].to(device) #features
              y = batch[1].to(device) #y labels
              logits = chosen_model(x)

              #calculate the loss
              validationbatch_loss = criterion(logits, y)
              validation_epoch_loss += (validationbatch_loss.item() * x.size(0))
              #calculate the accuracy per batch
              preds = torch.argmax(logits, dim=1)               
              validation_epoch_accuracy += ((preds == y).sum() / x.size(0))
                              
        #calculate the accuracy and loss on the validation data
        current_validation_epoch_loss = validation_epoch_loss/len(va_loader.dataset)
        validation_loss.append(current_validation_epoch_loss)
        current_validation_epoch_accuracy = validation_epoch_accuracy / len(va_loader)
        validation_accuracy.append(current_validation_epoch_accuracy)


        #print out the results at the end of each epoch
        print(f'Epoch {epoch}, training Loss { current_training_epoch_loss}, training Accuracy {current_epoch_accuracy}, validation loss {current_validation_epoch_loss}, validation accuracy {current_validation_epoch_accuracy}')

    #return the accuracy and loss lists
    return training_loss, training_accuracy, validation_loss, validation_accuracy
  

  def load_model_dataset(self, batch_size):
    #this will load the current working directory
    repository_root_directory = os.path.dirname(os.getcwd())
    rrd = "repository_root_directory:\t"
    print(rrd, repository_root_directory)

    if repository_root_directory not in sys.path:
        sys.path.append(repository_root_directory)
        print(rrd, "added to path")
    else:  
        print(rrd, "already in path")

    #extract the features from the dataset and save them as features and labels
    from features_extractor import FeaturesExtractor
    
    preprocessed_dataset = "../_02_data_preprocessed"
    extractor = FeaturesExtractor()
    data = extractor.extract_features_all_files(preprocessed_dataset)

    features = torch.tensor(data.drop('Genre', axis=1).values, dtype=torch.float32)
    labels = torch.tensor(data['Genre'].values, dtype=torch.long)

    #create tensor dataset
    from torch.utils.data import TensorDataset
    tensor_data = TensorDataset(features, labels)

    
    #split validation into validation and test data
    data_tr, data_te, data_va = torch.utils.data.random_split(tensor_data, [.7, .15, .15])

    #create the dataloaders
    bSize = batch_size
    
    train_loader = torch.utils.data.DataLoader(data_tr, batch_size=bSize, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(data_va, batch_size=len(data_va), shuffle=False)
    test_loader = torch.utils.data.DataLoader(data_te, batch_size=len(data_te), shuffle=False)

    #get the shape of each dataloader
    for batch in train_loader:
        size_of_train = len(train_loader.dataset)
        feature_shape = batch[0].shape
        label_shape = batch[1].shape
        train_loader_info = f"[{size_of_train},{feature_shape},  {label_shape}]"
        break
        

    for batch in valid_loader:
        feature_shape = batch[0].shape
        label_shape = batch[1].shape
        valid_loader_info = f"[{feature_shape},  {label_shape}]"
        break

    for batch in test_loader:
        feature_shape = batch[0].shape
        label_shape = batch[1].shape
        test_loader_info = f"[{feature_shape},  {label_shape}]"
        break

    #print out the dataset info
    print("shape of train: ", train_loader_info, "shape of validation: ", valid_loader_info, "shape of test: ", test_loader_info)

    #return the dataloaders
    return train_loader, valid_loader, test_loader