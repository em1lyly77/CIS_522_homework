import torch
from typing import Callable
import torch


class MLP(torch.nn.Module):
    """
    inherits from torch.nn.Module

    methods:
        __init__(): constructor that initializes the architecture
        forward(): makes a forward pass with the input data
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super().__init__()
        self.actv = activation()
        self.weight = initializer(torch.empty(input_size, hidden_size, num_classes))

        # Initialize layers of MLP
        self.layers = torch.nn.ModuleList()

        # create layers one by one
        for i in range(hidden_count):
            self.layers += [torch.nn.Linear(input_size, hidden_size, bias=True)]
            input_size = hidden_size
            initializer(self.layers[i].weight)

        # create output layer
        self.out = torch.nn.Linear(hidden_size, num_classes, bias=True)
        initializer(self.out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        for layer in self.layers:
            x = self.actv(layer(x))

        x = self.out(x)

        return x
