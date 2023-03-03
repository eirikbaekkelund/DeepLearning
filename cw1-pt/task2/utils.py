import os
import warnings
# avoid user warning from torchvision
with warnings . catch_warnings (): 
    warnings . simplefilter ( "ignore" ) 
    from PIL import Image

from torchvision import transforms
import torchvision
from classes import MixUp
from torch.utils.data import DataLoader

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
        assert montage_type in ['train', 'test'], "Montage type must be 'train' or 'test'"
        
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

def load_cifar10_data(sampling_method, batch_size=128, n_samples=60000, proportion_train=0.8):
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
        
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
        
      
        train_dataset.data = train_dataset.data[:int(n_samples*proportion_train)]
        train_dataset.targets = train_dataset.targets[:int(n_samples*proportion_train)]
        
        test_dataset.data = test_dataset.data[:int(n_samples*(1-proportion_train))]
        test_dataset.targets = test_dataset.targets[:int(n_samples*(1-proportion_train))]

        # add mixup on training data when passed into loader
        trainset_mixup = MixUp(train_dataset, sampling_method=sampling_method)

        # create a loader to get the a batch of 16 images
        train_loader = DataLoader(trainset_mixup, batch_size=16, shuffle=True, num_workers=8)
        # get a batch of 16 images
        img_train, _,_,_ = next(iter(train_loader))
        # create a montage of the images to be passed to training
        montage_mixup_png(img_train, 'train')

        # normalize train and test data
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ])
        
        train_dataset.transform = transform
        test_dataset.transform = transform

        # TODO For user: Ensure number of workers runs on your machine
        train_loader = DataLoader(trainset_mixup, batch_size=batch_size, shuffle=True, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        return train_loader, test_loader
