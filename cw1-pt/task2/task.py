import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from PIL import Image
    import os
    import torch.nn.functional as F

    ######### HELPER FUNCTIONS #########
    
    
    def montage_mixup_png(data, montage_type = 'train'):
        """ 
        Creates a montage of the images in the data.
        Args:
            data (torch array): Data to create montage from.
            n_rows (int): Number of rows in the montage.
            n_cols (int): Number of columns in the montage.
        Returns:
            PIL image: Montage of the images in the data.
        """
        assert montage_type in['train', 'test'], "Montage type must be 'train' or 'test'"
        
        if montage_type == 'train':
            n_rows = 4
            n_cols = 4
            file_name = '/mixup.png'
        
        else:
            n_rows = 6
            n_cols = 6
            file_name = '/result.png'
        
        assert data.shape[0] == n_rows * n_cols, "Number of images must be equal to n_rows * n_cols"

        # Create a new image with a size that can fit all the images
        new_im = Image.new('RGB', (data.shape[2] * n_cols, data.shape[3] * n_rows))
    
        # Paste the images into the new image
        for i in range(n_rows):
            for j in range(n_cols):
                # get image
                im = data[i*n_cols + j, :, :, :]
                # im = im.permute(1, 2, 0)
                im = transforms.ToPILImage()(im)
                new_im.paste(im, (j * data.shape[2], i * data.shape[3]))
        
        # Save the new image within the same folder as running file
        path = os.path.dirname(os.path.abspath(__file__))
        new_im.save( path + file_name)
        
        return new_im
    
    ######### MIXUP CLASS #########

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
            # get images
            x, y = self.dataset[index]
            
            # TODO might need to be integer
            idx = torch.randperm(len(self.dataset))[0]

            x2, y2 = self.dataset[idx]

            lam = self.sample_func.sample()

            x_mix = lam * x + (1 - lam) * x2
            y_mix = lam * y + (1 - lam) * y2
            
            return x_mix, y_mix.float(), y2.float(), lam
        
        def __len__(self):
            return len(self.dataset)
    
    ######### RESNET50 CLASS #########
    
    class ResNet(nn.Module):
        """ 
        Mixup class that inherits from torch.nn.Module.
        Args:
            sampling_method (int): Sampling method to use. 1 for Beta distribution and 2 for uniform distribution.
            resnet (torchvision model): Resnet model to use.
        """
        def __init__(self):
            super(ResNet, self).__init__()       
            self.resnet50 = torchvision.models.resnet50(pretrained=True)

            # replace last FC layer and add softmax layer
            self.resnet50.fc = nn.Sequential(
                nn.Linear(2048, 10),
                nn.Softmax(dim=1)
            )
            # set last layer to train and gradient to true
            self.resnet50.fc[0].requires_grad = True
            self.resnet50.train()

            self.criterion = nn.CrossEntropyLoss(reduction='none')
            self.optimizer = torch.optim.SGD(self.resnet50.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
        def mixup_loss(self, pred, y, y_inferred, lam):
            """ 
            Calculates the loss for mixup.
            Args:
                pred (torch array): Predictions.
                y (torch array): Label with mix.
                y_inferred (torch array):  Label mixed into y.
                lam (float): Lambda value.
            Returns:
                torch array: Loss.

            """
            return torch.sum(lam * self.criterion(pred, y) + (1 - lam) * self.criterion(pred, y_inferred))
        
        def train(self, train_loader, device, n_epochs=10):
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
            print('==> Starting training...')
            
            self.resnet50.train()
            create_montage = True
            
            for epoch in range(n_epochs):
                for data, target, target_mix, lam in train_loader:

                    x_mix, y, y_infer, lam = data.to(device), target.to(device), target_mix.to(device), lam.to(device)        
                    
                    if create_montage:
                        create_montage = False
                        montage_mixup_png(x_mix[:16], 'train')

                    x_mix = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x_mix)
                    pred = self.resnet50(x_mix)
                    loss = self.mixup_loss(pred, y, y_infer, lam)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                print('Epoch: {} \tLoss: {:.6f}'.format(epoch + 1, loss.item()))
                print('predictions: ', pred[0, :])
                print('target: ', target[0, :])
                print('\n')

            print('==> Finished training...')

        def test(self, test_loader, device):
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
            
            create_montage = True
            
            with torch.no_grad():
                
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = self.resnet50(data)
                    test_loss += torch.sum(self.criterion(output, target.float())).item()  # sum up batch loss
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    target = target.argmax(dim=1, keepdim=True)
                    correct = torch.mean((pred.squeeze() == target.squeeze()).float())
                    
                    if create_montage:
                        # should unnormalize data using transform in pytorch
                        montage_mixup_png(data[:36], 'test')
                        for i in range(36):
                            print("| Target \t | Predicted")
                            print("| {} \t | {}".format(target[i], pred[i]))
                        create_montage = False

            test_loss /= len(test_loader.dataset)

            print(f'Test set: Average loss: {test_loss}')
            print(f'Accuracy: {correct}/{len(test_loader.dataset)}')
            print('==> Finished testing...')

    ######### DATA HELPER #########

    def load_cifar10_data(sampling_method, batch_size=128, n_samples=1000, proportion_train=0.8):
        """ 
        Loads the data from the MixUp class.

        Args:
            sampling_method (int): Sampling method to use. 1 for Beta distribution and 2 for uniform distribution.
            batch_size (int): Batch size to use.
            n_samples (int): Number of samples to use.
            proportion_train (float): Proportion of samples to use for training.
        Returns:
            torch.utils.data.DataLoader: Training data loader.
            torch.utils.data.DataLoader: Test data loader.
        """
        # load data
        transform = transforms.Compose([transforms.ToTensor()])
        
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        # onehot encode targets
        train_dataset.targets = F.one_hot(torch.tensor(train_dataset.targets), num_classes=10)
        test_dataset.targets = F.one_hot(torch.tensor(test_dataset.targets), num_classes=10)

        train_dataset.data = train_dataset.data[:int(n_samples*proportion_train)]
        train_dataset.targets = train_dataset.targets[:int(n_samples*proportion_train)]
        
        test_dataset.data = test_dataset.data[:int(n_samples*(1-proportion_train))]
        test_dataset.targets = test_dataset.targets[:int(n_samples*(1-proportion_train))]


        trainset_mixup = MixUp(train_dataset, sampling_method=sampling_method) 

        # normalize train and test data
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        test_dataset.transform= transform

        # TODO For user: Ensure number of workers runs on your machine
        train_loader = DataLoader(trainset_mixup, batch_size=batch_size, shuffle=True, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        return train_loader, test_loader

    if __name__ == "__main__":
        print('\n')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)      
        print('\n')
        for method in [1, 2]:

            train_loader, test_loader = load_cifar10_data(sampling_method=method, batch_size=128, n_samples=1000, proportion_train=0.8)
            resnet = ResNet().to(device)
            resnet.train(train_loader=train_loader, device=device) 
            print('\n')

            resnet.test(test_loader=test_loader, device=device)
            print('\n')
            
            # Save model in same folder as file
            torch.save(resnet.resnet50.state_dict(), f"{os.path.dirname(os.path.abspath(__file__))}/model_{method}.pt")
