"""
Author: Hao Zheng
Matr.Nr.: K01608113
Exercise 5
"""

'''' 
Modified version of the architecture of the example project: For my model, I used the provided model architecture of the examples project and adjusted it to our challenge
The input channels are 4( RGB values and the known array) and output channels are 3( only predicted RGB for whole images)
The activation function is the ReLU function.

CNN with: 5 hidden layers, 64 kernels, kernel size 3
'''
import torch
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class myCNN(torch.nn.Module):
    def __init__(self, n_input_channels: int = 4, n_hidden_layers: int = 5, n_hidden_kernels: int = 64,
                 kernel_size: int = 3):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyperparameters"""
        super().__init__()

        # conv hidden layers

        hidden_layers = []

        for i in range(n_hidden_layers):
            hidden_layers.append(torch.nn.Conv2d(
                in_channels=n_input_channels,
                out_channels=n_hidden_kernels,
                kernel_size=kernel_size,
                padding=int(kernel_size / 2)
            ))
            # use ReLU as activation function - good for images

            hidden_layers.append(torch.nn.ReLU())
            n_input_channels = n_hidden_kernels
        self.hidden_layers = torch.nn.Sequential(*hidden_layers)
        # conv output layer
        self.output_layer = torch.nn.Conv2d(
            in_channels=n_input_channels,
            out_channels=3,
            kernel_size=kernel_size,
            padding=int(kernel_size / 2)
        )

    def forward(self, x):
        """Apply CNN to input `x` of shape (N, n_channels, X, Y), where N=n_samples and X, Y are spatial dimensions"""
        # used to specify the “flow” through the architecture, i.e., how a given input is transformed into the module’s output)
        # The.forward() method will be executed when your class instance is applied to an input
        cnn_out = self.hidden_layers(x)  # apply hidden layers (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        pred = self.output_layer(cnn_out)  # apply output layer (N, n_kernels, X, Y) -> (N, 1, X, Y)
        return pred
