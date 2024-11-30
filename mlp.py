import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler
import sys
import os

class MLP(nn.Module):
  def __init__(self, in_dim, out_dim, hid_layers):
    super(MLP, self).__init__()
    self.nnLayers = [nn.Flatten()] #this will hold all the layers of the neural netowrk. Flatten is to make sure that the shapes are all the same to avoid any mismatches. 
    #input_dim = in_dim
    for i in range (len(hid_layers)):
      hidden_dim = hid_layers[i]
      if i == 0: #first input layer. This takes the specified input dimension and the dimension of first hidden layer
        self.nnLayers.append(nn.Linear(in_dim, hidden_dim))
        
      else: #the hidden layers.
        self.nnLayers.append(nn.Linear(input_dim, hidden_dim))
      #this will connect the layers together by storing the dimension of the output from the layer for the next layer.
      input_dim = hidden_dim
      #apply the relu activation function
      self.nnLayers.append(nn.ReLU())
      
    #output layer  
    self.nnLayers.append(nn.Linear(input_dim, out_dim))

    self.net = nn.Sequential(*self.nnLayers)


  def forward(self, x):
    #flatten the input
    return self.net(x)
  
  def test_model(self, model, criterion, loader, device):
    chosenModel = model
    chosenCriterion = criterion
    chosenLoader = loader
    chosenDevice = device
    
    #store the values of each batch in these variables
    correctPredictions = 0
    totalSamples = 0
    totalLoss = 0.0

    #set the model to evaluation mode
    chosenModel.eval()
    #disable gradient computation
    with torch.no_grad():
        for batch in chosenLoader:
            #first get the batch data
            x = batch[0].to(chosenDevice) # input data(features)
            y = batch[1].to(chosenDevice) # target labels that corresponds with the input data
            logits = chosenModel(x)

            #get the predictions so the accuracy can be calculated
            preds = torch.argmax(logits, dim=1)
            #calculate the accuracy of the current batch. .sum counts the number of correct predictions for the tensor (preds == y). That looks at all the predictions in the batch against the correct label. 
            #add up all the total number of correct predictions per batch. This works because true predictions are value 1 and false are 0 so only 1's add up.
            correctPredictions += ((preds == y).sum().item())
            #calculate the loss. The outpus and the labels
            batchLoss = chosenCriterion(logits,y)
            
            totalLoss += (batchLoss.item() * x.size(0))

            #this gets the total number of samples in this batch and adds it to the total number of samples. 
            totalSamples += x.size(0)

    #calculate the average accuracy accross all the batches by comparing the number of correct predictions to the total number of samples.        
    averageAccuracy = correctPredictions / totalSamples

    #calculate the average loss
    averageLoss = totalLoss / totalSamples
    
    return averageAccuracy, averageLoss
  
  def train_model(self, model, criterion, optimizer, tr_loader, va_loader, n_epochs, device):
    chosenModel = model
    
    numberOfEpochs = n_epochs
    trainingLoss = []
    trainingAccuracy = []
    validationLoss = []
    validationAccuracy = []
    #go through each epoch
    for epoch in range(numberOfEpochs):
        #set the model to train mode
        chosenModel.train()

        #for training data
        epochLoss = 0
        epochAccuracy = 0

        #for validation data
        validationEpochLoss = 0
        validationEpochAccuracy = 0
        

        #go through all the training minibatches
        for i, batch in enumerate(tr_loader): 
            #retrieve the data
            x = batch[0].to(device)
            y = batch[1].to(device)
            #perform the forward pass
            logits = chosenModel(x)
            #compute the loss of this batch
            loss = criterion(logits, y)
            epochLoss += (loss.item()* x.size(0))
            #compute the accuracy of the current batch
            preds = torch.argmax(logits, dim=1)
            epochAccuracy += ((preds == y).sum() / x.size(0))
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #calculate the accuracy and loss on the training data
        currentTrainingEpochLoss = epochLoss/len(tr_loader.dataset)
        trainingLoss.append(currentTrainingEpochLoss)
        currentEpochAccuracy = epochAccuracy/len(tr_loader)
        trainingAccuracy.append(currentEpochAccuracy)

        chosenModel.eval()
        with torch.no_grad():
            
            #go through the batches of validation data
            for i,batch in enumerate(va_loader):
                x = batch[0].to(device) #features
                y = batch[1].to(device) #y labels
                logits = chosenModel(x)

                #calculate the loss
                validationBatchLoss = criterion(logits, y)
                validationEpochLoss += (validationBatchLoss.item() * x.size(0))
                #calculate the accuracy per batch
                preds = torch.argmax(logits, dim=1)               
                validationEpochAccuracy += ((preds == y).sum() / x.size(0))
                

                
        #calculate the accuracy and loss on the validation data
        currentValidationEpochLoss = validationEpochLoss/len(va_loader.dataset)
        validationLoss.append(currentValidationEpochLoss)
        currentValidationEpochAccuracy = validationEpochAccuracy / len(va_loader)
        validationAccuracy.append(currentValidationEpochAccuracy)


        #print out the results at the end of each epoch
        print(f'Epoch {epoch}, training Loss { currentTrainingEpochLoss}, training Accuracy {currentEpochAccuracy}, validation loss {currentValidationEpochLoss}, validation accuracy {currentValidationEpochAccuracy}')

    #return the accuracy and loss lists
    return trainingLoss, trainingAccuracy, validationLoss, validationAccuracy
  
  def load_model_dataset(self, batch_size):
    #download
    #dataTr = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    #dataVa = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    repository_root_directory = os.path.dirname(os.getcwd())
    rrd = "repository_root_directory:\t"
    print(rrd, repository_root_directory)

    if repository_root_directory not in sys.path:
        sys.path.append(repository_root_directory)
        print(rrd, "added to path")
    else:  
        print(rrd, "already in path")

    from features_extractor import FeaturesExtractor
    import numpy as np


    preprocessed_dataset = "../_02_data_preprocessed"
    extractor = FeaturesExtractor()
    data = extractor.extract_features_all_files(preprocessed_dataset)

    features = torch.tensor(data.drop('Genre', axis=1).values, dtype=torch.float32)
    labels = torch.tensor(data['Genre'].values, dtype=torch.long)

    #create tensordataset
    from torch.utils.data import TensorDataset
    tensorData = TensorDataset(features, labels)

    #features = np.array(data['features'])
    #labels = np.array(data['labels'])
    #print("Shape of the deata", features.shape)
    #print("Number of features: ",features.shape[1])
    #print("Number of datapoints: ", features.shape[0])

    #size of the subset
    #subSetSize = 10000

    #choose the random datapoints from the larger dataset
    #subsetIindices = torch.randperm(len(dataTr))[:subSetSize]
    #subSetDataTr = torch.utils.data.Subset(dataTr, subsetIindices)
    
    #split validation into validation and test data
    dataTr, dataTe, dataVa = torch.utils.data.random_split(tensorData, [.25, .25, .5])
    #check the classes of the data
    #print(dataTr.classes)
    
    #create the dataloaders
    bSize = batch_size
    
    trainLoader = torch.utils.data.DataLoader(dataTr, batch_size=bSize, shuffle=True)
    validLoader = torch.utils.data.DataLoader(dataVa, batch_size=len(dataVa), shuffle=False)
    testLoader = torch.utils.data.DataLoader(dataTe, batch_size=len(dataTe), shuffle=False)

    #get the shape of each dataloader
    for batch in trainLoader:
        sizeOfTrain = len(trainLoader.dataset)
        featureShape = batch[0].shape
        labelShape = batch[1].shape
        trainLoaderInfo = f"[{sizeOfTrain},{featureShape},  {labelShape}]"
        break
        

    for batch in validLoader:
        featureShape = batch[0].shape
        labelShape = batch[1].shape
        validLoaderInfo = f"[{featureShape},  {labelShape}]"
        break

    for batch in testLoader:
        featureShape = batch[0].shape
        labelShape = batch[1].shape
        testLoaderInfo = f"[{featureShape},  {labelShape}]"
        break

    #print out the dataset info
    print("shape of train: ", trainLoaderInfo, "shape of validation: ", validLoaderInfo, "shape of test: ", testLoaderInfo)

    #return the dataloaders
    return trainLoader, validLoader, testLoader