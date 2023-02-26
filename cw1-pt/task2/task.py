import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    # import os
    # os.system('sudo ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so.8 /usr/lib/x86_64-linux-gnu/libjpeg.so.62')
    from torchvision.models import resnet50 as resnet
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms

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
            self.loss = nn.functional.cross_entropy

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
            # y_hat = lam * y + (1 - lam) * y[idx, :]
            y_hat = lam * y + (1 - lam) * y.gather(0, idx.expand(-1, y.size(1), -1, -1))

            
            return x_hat, y_hat
        
        
        
    def train(model, train_loader, optimizer, device):
        """ 
        Trains the model on the training data.
        Args:
            model (torchvision model): Model to train.
            train_loader (torch.utils.data.DataLoader): Training data loader.
            optimizer (torch.optim): Optimizer to use.
            device (torch.device): Device to use.
        """
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            print('data shape: ', data.shape)
            print('target shape: ', target.shape)
            print(target)
            # Perform mixup
            data, target = model(data, target)
            
            optimizer.zero_grad()
            output = model.model(data)
            loss = model.loss(output, target)
            loss.backward()
            optimizer.step()

    def test(model, test_loader, device):
        """ 
        Tests the model on the test data.
        Args:
            model (torchvision model): Model to test.
            test_loader (torch.utils.data.DataLoader): Test data loader.
            device (torch.device): Device to use.
        """
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model.model(data)
                test_loss += model.loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%'')'.format(test_loss, correct, len(test_loader.dataset)))

    def load_data(num_pics=16):
        """ 
        Loads the training and test data.
        Args:
            num_pics (int): Number of pictures to load.
        Returns:
            torch.utils.data.DataLoader: Training data loader.
            torch.utils.data.DataLoader: Test data loader.
        """

        print('==> Preparing data...')

        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        
        trainset.data = trainset.data[:num_pics]
        trainset.targets = trainset.targets[:num_pics]
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
        
        testset.data = testset.data[:num_pics]
        testset.targets = testset.targets[:num_pics]
        
        testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                shuffle=False, num_workers=2)
        
        return trainloader, testloader

    def fix_libjpeg8():
        """ 
        Fixes a bug in the libjpeg8 library.
        """
        import os
        os.system('sudo ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so.8 /usr/lib/x86_64-linux-gnu/libjpeg.so.62')

    if __name__ == "__main__":
        # Fix libjpeg8 bug
        fix_libjpeg8()
        print('==> Starting training...')
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        # Load data
        train_loader, test_loader = load_data()
        print('Number of training samples:', len(train_loader.dataset))
        # Create model
        model = mixup(sampling_method=1).to(device)
        
        # Create optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        # Train model
        for epoch in range(1, 10 + 1):
            train(model, train_loader, optimizer, device)
            test(model, test_loader, device)
            
        # Save model
        torch.save(model.state_dict(), "model.pt")