import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)

    from torchvision.models import resnet50 
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    from PIL import Image
    import os

    def load_CIFAR10_data(n_samples=1000, prop_train=0.8):
        """ 
        Loads the training and test data.
        Args:
            num_pics (int): Number of pictures to load.
        Returns:
            torch.utils.data.DataLoader: Training data loader.
            torch.utils.data.DataLoader: Test data loader.
        """
        
        print('==> Preparing data...')

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        
        n_train = int(n_samples*prop_train)
        n_test = int(n_samples*(1 - prop_train) + 1)

        trainset.data = trainset.data[:n_train]
        trainset.targets = trainset.targets[:n_train]

        testset.data = testset.data[:n_test]
        testset.targets = testset.targets[:n_test]
        # one hot encode targets
        trainset.targets = torch.nn.functional.one_hot(torch.tensor(trainset.targets), num_classes=10)
        testset.targets = torch.nn.functional.one_hot(torch.tensor(testset.targets), num_classes=10)
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=36, shuffle=False, num_workers=2)
        
        print('==> Done!')

        return trainloader, test_loader
    
    def montage_mixup_png(data, n_rows=4, n_cols=4):
        """ 
        Creates a montage of the images in the data.
        Args:
            data (torch array): Data to create montage from.
            n_rows (int): Number of rows in the montage.
            n_cols (int): Number of columns in the montage.
        Returns:
            PIL image: Montage of the images in the data.
        """
        assert data.shape[0] == n_rows * n_cols, "Number of images must be equal to n_rows * n_cols"
        
        # Create a new image with a size that can fit all the images
        new_im = Image.new('RGB', (data.shape[2] * n_cols, data.shape[3] * n_rows))
    
        # Paste the images into the new image
        for i in range(n_rows):
            for j in range(n_cols):
                # get image
                im = data[i*4 + j, :, :, :]
                # im = im.permute(1, 2, 0)
                im = transforms.ToPILImage()(im)
                new_im.paste(im, (j * data.shape[2], i * data.shape[3]))
        
        # Save the new image within the same folder as running file
        path = os.path.dirname(os.path.abspath(__file__))
        new_im.save( path + '/mixup.png')
        return new_im
    
    class MixUp(torch.nn.Module):
        """ 
        Mixup class that inherits from torch.nn.Module.
        Args:
            sampling_method (int): Sampling method to use. 1 for Beta distribution and 2 for uniform distribution.
            resnet (torchvision model): Resnet model to use.
        """
        def __init__(self, sampling_method, resnet50=resnet50):
            super(MixUp, self).__init__()
            assert sampling_method in [1, 2], "Sampling method must be 1 or 2"
            
            if sampling_method == 1:
                alpha = torch.tensor(1.0, dtype=torch.float32)
                self.sample_func = torch.distributions.Beta(alpha, alpha)
            else:
                uniform_range = torch.arange(0, 1, 0.01)
                self.sample_func = torch.distributions.Uniform(uniform_range[0], uniform_range[-1])
            
            self.network = resnet50(pretrained=False, progress=True)
            # replace last FC layer and add softmax layer
            self.network.fc = nn.Sequential(
                nn.Linear(2048, 10),
                nn.Softmax(dim=1)
            )

            for name, params in self.network.named_parameters():
            
                if name.startswith('fc'):
                    params.requires_grad = True
            
                else:
                    params.requires_grad = True
            
            self.loss = nn.functional.cross_entropy

        def forward_mixup(self, x, y):
            """ 
            Performs mixup algorithm on the input data and labels.            

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
            x_hat = lam * x + (1 - lam) * x[idx, :, :, :]
            y_hat = torch.tensor(lam * y + (1 - lam) * y[idx], dtype= torch.float32)
        
            return x_hat, y_hat
        
    def train(sampling_method, train_loader, device, n_epochs=10, lr=0.001, momentum=0.01):
        """ 
        Trains the model on the training data.
        
        Args:
            model (torchvision model): Model to train.
            train_loader (torch.utils.data.DataLoader): Training data loader.
            optimizer (torch.optim): Optimizer to use.
            device (torch.device): Device to use.
        Returns:
            torchvision model: Trained model.
        """

        model = MixUp(sampling_method)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        
        print('==> Starting training...')

        for epoch in range(n_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):

                data, target = data.to(device), target.to(device)        
                data, target = model.forward_mixup(data, target)
                
                if epoch == 0 and batch_idx == 0:
                    montage_mixup_png(data)

                optimizer.zero_grad()
                output = model.network.forward(data)
                loss = model.loss(output, target)
                loss.backward()
                optimizer.step()
            
#            print('gradients: ', model.network.fc[0].weight.grad[0, 0:5])   
            print('Epoch: {} \tLoss: {:.6f}'.format(epoch + 1, loss.item()))
        
        print('==> Finished training...')
        
        return model

    def test(model, test_loader, device):
        """ 
        Tests the model on the test data.
        
        Args:
            model (torchvision model): Model to test.
            test_loader (torch.utils.data.DataLoader): Test data loader.
            device (torch.device): Device to use.
        """
        print('==> Testing model...')

        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model.network.forward(data)
                test_loss += model.loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                target = target.argmax(dim=1, keepdim=True)
                correct = torch.mean((pred.squeeze() == target.squeeze()).float())

        test_loss /= len(test_loader.dataset)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%'')'.format(test_loss, correct, len(test_loader.dataset)))
        print('==> Finished testing...')

    if __name__ == "__main__":

        print('\n')
                
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        
        print('\n')
        # Load data
        train_loader, test_loader = load_CIFAR10_data()

        print('\n')
        print('Number of training samples:', len(train_loader.dataset))
        print('Number of test samples:', len(test_loader.dataset))        
        
        print('\n')
        # Train model
        model = train(sampling_method=1, train_loader=train_loader,device=device)

        print('\n')

        test(model, test_loader, device)
        
        print('\n')
        # Save model
        torch.save(model.state_dict(), "model.pt")