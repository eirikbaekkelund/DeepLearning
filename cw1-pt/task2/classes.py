import torch.nn as nn
from torchvision import models

class CustomResNet50(nn.Module):
    """
    ResNet50 model with custom fully connected layer.
    """
    def __init__(self):
        super(CustomResNet50, self).__init__()
        self.resnet50 = models.resnet50(weights=None)
        # remove original fully connected layer
        self.resnet50.fc = nn.Identity()
        # add new fully connected layer
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        """
        Computes forward pass of model.
        """
        x = self.resnet50(x)
        # reshape output tensor to match input size of fc layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
