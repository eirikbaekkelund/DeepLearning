 ######### MIXUP CLASS #########

import torch
import torch.nn as nn
from torchvision import models

class MixUp(torch.utils.data.Dataset):
    """
    Mixup class that inherits from torch.utils.data.Dataset.
    Args:
        dataset (torch.utils.data.Dataset): Dataset to use.
        sampling_method (int): Sampling method to use. 1 for Beta distribution and 2 for uniform distribution.
    """
    def __init__(self, dataset, sampling_method):
        
        super(MixUp, self).__init__()

        assert sampling_method in [1,2], "Sampling method must be 1 or 2"
        
        if sampling_method == 1:
            alpha = torch.tensor(1, dtype=torch.float32)
            self.sample_func = torch.distributions.Beta(alpha, alpha)
        
        else:
            uniform_range = torch.arange(0, 1, 0.01)
            self.sample_func = torch.distributions.Uniform(uniform_range[0], uniform_range[-1])

        self.dataset = dataset
    
    def __getitem__(self, index):
        """ 
        Mix up algorithm to be used in the dataloader.
        """
        # get images
        x, y = self.dataset[index]
        
        # TODO might need to be integer
        idx = torch.randperm(len(self.dataset))[0]

        x2, y2 = self.dataset[idx]

        lam = self.sample_func.sample()

        x_mix = lam * x + (1 - lam) * x2
        y_mix = lam * y + (1 - lam) * y2
        
        return x_mix, y_mix, y2, lam
    
    def __len__(self):
        return len(self.dataset)

class CustomResNet50(nn.Module):
    def __init__(self):
        super(CustomResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=False)
        # remove original fully connected layer
        self.resnet50.fc = nn.Identity()
        # add new fully connected layer
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.resnet50(x)
        # reshape output tensor to match input size of fc layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
