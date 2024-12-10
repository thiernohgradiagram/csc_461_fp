import sys
import os

repository_root_directory = os.path.dirname(os.getcwd())
rrd = "repository_root_directory:\t"
print(rrd, repository_root_directory)

if repository_root_directory not in sys.path:
    sys.path.append(repository_root_directory)
    print(rrd, "added to path")
else:  
    print(rrd, "already in path")


import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_layers, dropout_rate):
        """
        Initializes the Multilayer Perceptron (MLP) network.

        Args:
            in_dim (int): Input dimension - number of features in the input data.
            out_dim (int): Output dimension - number of output neurons.
            hid_layers (list): List of integers specifying the number of neurons in each hidden layer.
        """
        super(MLP, self).__init__()                                             # Calling the constructor of nn.Module to properly initialize the model by inheriting the good stuff

        neural_network = []                                                     # we start with an empty neural network with zero layers
        current_dim = in_dim                                                    # the number of neurons of each layers - initially set to in_dim

        # Add batch normalization for the input layer
        neural_network.append(nn.BatchNorm1d(current_dim))

        # Sequentially Adding hidden layers with ReLU activation to the neural network
        for h_dim in hid_layers:
            # Assume v is the vector representing the previous layer
            v_transformed = nn.Linear(current_dim, h_dim)                       # applying a linear transformation and storing the transformed vector v
            neural_network.append(nn.BatchNorm1d(h_dim))  # Add batch normalization
            v_transformed_normalized = nn.ReLU()                                # applying the Rectified Linear Unit function to the vector v_transformed to normalize it
            neural_network.append(v_transformed)                                # adding v_transformed to the neural network
            neural_network.append(v_transformed_normalized)                     # adding v_transformed_normalized to the neural network
            neural_network.append(nn.Dropout(dropout_rate))
            current_dim = h_dim                                                 # updating current_dim to reflect the current layer being processed

        # Add the output layer (linear activation)                              # no non-linear activation function applied
        output_vector = nn.Linear(current_dim, out_dim)                         # applying a linear transformation to get the output vector
        neural_network.append(output_vector)                                    # adding the non-normalized vector to the neural network

        self.network = nn.Sequential(*neural_network)                           # Create the sequential neural network model


    def forward(self, x):
        """
        Performs the forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor (mini-batch).

        Returns:
            torch.Tensor: Output of the network - the predicted output vectors of the model.
        """
        #x = x.view(x.size(0), -1)  # Flatten the input tensor
        return self.network(x)
    