import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    custom neural network class that inherits nn.Module and quickly classify images
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        # self.size_after_conv = (self.num_channels + 2*1 - 3)/1 +1
        # self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.conv1 = nn.Conv2d(num_channels, 16, 5, stride=2, padding=1)

        # self.conv2 = nn.Conv2d(16, 16, 3)  # 3 or 5

        self.maxpool = nn.MaxPool2d(3, 3)

        self.batchnorm = nn.BatchNorm2d(16)

        self.fc = nn.Linear(16 * 5 * 5, num_classes, True)  # 3 then 6, 5 then 6

        # self.network = nn.Sequential(
        #     nn.Conv2d(num_channels, 6, 5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(6, 16, 5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),
        #     nn.Flatten(),
        #     nn.Linear(16*5*5, num_classes, True)
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass of the input data thru the custom network
        """
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.maxpool(x)

        # x = self.conv2(x)
        # x = F.relu(x)
        # x = self.maxpool(x)

        x = x.view(-1, 16 * 5 * 5)
        x = self.fc(x)
        # y = self.network(x)
        return x
