# import warnings
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=UserWarning)

from torchvision.models import resnet50 as resnet
import torch

# Creating a class mixup inheriting
class mixup(torch.nn.Module):
    """ 
    Mixup class that inherits from torch.nn.Module.
    Args:
        sampling_method (int): Sampling method to use. 1 for Beta distribution and 2 for uniform distribution.
        resnet (torchvision model): Resnet model to use.
    """
    def __init__(self, sampling_method, resnet=resnet):
        super(mixup, self).__init__()
        assert sampling_method in [1, 2], "Sampling method must be 1 or 2"
        
        if sampling_method == 1:
            alpha = torch.tensor(1.0, dtype=torch.float32)
            self.sample_func = torch.distributions.Beta(alpha, alpha)
        else:
            uniform_range = torch.arange(0, 1, 0.01)
            self.sample_func = torch.distributions.Uniform(uniform_range[0], uniform_range[-1])
        
        self.model = resnet(pretrained=True)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        """ 
        Performs mixup on the input data and labels.
        Args:
            x (torch array): Input data.
            y (torch array): Input labels.
        Returns:
            torch array: Mixed up data.
            torch array: Mixed up labels.
        """
        # Generate a random number lambda
        lam = self.sample_func.sample()
        
        # Generate a random permutation of the batch
        idx = torch.randperm(x.size()[0])
        
        # Perform mixup
        x_hat = lam * x + (1 - lam) * x[idx, :]
        y_hat = lam * y + (1 - lam) * y[idx, :]
        
        return x_hat, y_hat
    
    def train(self, x, y, n_epochs=10, batch_size=32, lr=0.001, print_loss=False):
        # Define the loss function and the optimizer
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # Train the model
        for epoch in range(n_epochs):
            # Generate a random permutation of the batch
            idx = torch.randperm(x.size()[0])
            # Split the data into batches
            x_batches = torch.split(x[idx, :], batch_size)
            y_batches = torch.split(y[idx, :], batch_size)
            # Train the model on each batch
            for x_batch, y_batch in zip(x_batches, y_batches):
                # Forward pass
                y_pred = self.model(x_batch)
                loss = self.loss(y_pred, y_batch)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Print the loss
            if print_loss:
                print(f"Epoch {epoch}: Loss = {loss.item()}")
        return self.model