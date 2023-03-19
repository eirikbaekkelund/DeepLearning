from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split

def load_cifar10_train_val_test(batch_size=128):
    """ 
    Loads the CIFAR10 dataset from torchvision.

    Args:
        batch_size (int): Batch size to use.
    Returns:
        torch.utils.data.DataLoader: Training data loader.
        torch.utils.data.DataLoader: Validation data loader.
        torch.utils.data.DataLoader: Test data loader.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load the dataset and split into train, validation and test sets
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    n_samples = len(dataset)
    n_train = int(0.8 * n_samples)
    n_dev = int(0.1 * n_train)
    n_train = n_train - n_dev
    n_test = n_samples - n_train - n_dev
    train_dataset, dev_dataset, test_dataset = random_split(dataset, [n_train, n_dev, n_test])

    # Create data loaders for the train, validation and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, dev_loader, test_loader

