from utils import load_cifar10_data, montage_mixup_png
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from classes import CustomResNet50
from torch.autograd import Variable
# import transforms from torch
from torchvision import transforms 
import os

class MixUp(nn.Module):
    """ 
    Task that uses mixup and SGD for classification
    Using one of two sampling methods for mixup.
    They come implicity from the MixUp class applied to the training data.
    
    Args:
        alpha (float): Alpha value for Beta distribution.
        optimizer (str): Optimizer to use. 'sgd' or 'adam'.
        method (int): Sampling method to use. 1 for Beta distribution and 2 for uniform distribution.
        criterion (torch loss function): Loss function to use.
    """
    def __init__(self, method=1, optimizer='sgd',alpha=0.2, criterion=CrossEntropyLoss()):
        assert optimizer in ['sgd', 'adam'], "Optimizer must be 'sgd' or 'adam'."
        assert method in [1, 2], "Method must be 1 or 2. 1 for Beta distribution and 2 for uniform distribution."
        assert alpha > 0, "Alpha must be greater than 0."
        assert isinstance(criterion, torch.nn.modules.loss._Loss), "Criterion must be a torch loss function."
        
        super(MixUp, self).__init__()       
        self.resnet50 = CustomResNet50()
        self.alpha = alpha
        self.method = method
        self.criterion = criterion

        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.resnet50.parameters(), lr=1e-3, momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(self.resnet50.parameters(), lr=1e-3)
    
    def mix_up(self, x, y):
        """
        Perform mix up algorithm on the data.

        Args:
            x (torch array): Images.
            y (torch array): Labels.
            method (int): Sampling method to use. 1 for Beta distribution and 2 for uniform distribution.
        
        Returns:
            mixed_x (torch array): Mixed images.
            y_a (torch array): Labels for images.
            y_b (torch array): Labels for mixed images.

        """
        if self.method == 1:
            lam = torch.distributions.Beta(torch.tensor([self.alpha]), torch.tensor([self.alpha])).sample()
        else:
            lam = torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([1.0])).sample()
        
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def mixup_loss(self, predictions, y, y_inferred, lam):
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
        return lam * self.criterion(predictions, y) + (1 - lam) * self.criterion(predictions, y_inferred)
    
    def train_model(self, train_loader):
        """ 
        Trains the model on the training data.
        
        Args:
            model (torchvision model): Model to train.
            train_loader (torch.utils.data.DataLoader): Training data loader with mixup.
            optimizer (torch.optim): Optimizer to use.
        Returns:
            train_loss (float): Training loss.
            train_acc (float): Training accuracy.
        """
        self.resnet50.train()
        train_loss = 0
        correct = 0
        total = 0

        for data, target in train_loader:
            inputs, targets_a, targets_b, lam = self.mix_up(data, target)

            # maps data to autograd for backprop
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
            
            outputs = self.resnet50(inputs)
            
            loss = self.mixup_loss(predictions=outputs, 
                                   y=targets_a, 
                                   y_inferred=targets_b, 
                                   lam=lam)

            train_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (lam * predicted.eq(targets_a.data).cpu().sum().float() + 
                        (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        return train_loss, train_acc.item()
    
    def test_model(self, test_loader):
        """ 
        Tests the model on the test data.
        
        Args:
            model (torchvision model): Model to test.
            test_loader (torch.utils.data.DataLoader): Test data loader.
        """
        test_loss = 0
        correct = 0
        total = 0

        self.resnet50.eval()
        with torch.no_grad():
            for data, targets in test_loader:
                
                inputs, targets = Variable(data), Variable(targets)
                outputs = self.resnet50(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

            test_loss = test_loss /  len(test_loader)
            test_acc = 100. * (correct / total )
        
        return test_loss, test_acc.item()
    
    def train(self, train_loader, test_loader, epochs=10):
        """ 
        Trains and tests the model.
        
        Args:
            model (torchvision model): Model to train and test.
            train_loader (torch.utils.data.DataLoader): Training data loader with mixup.
            test_loader (torch.utils.data.DataLoader): Test data loader.
            optimizer (torch.optim): Optimizer to use.
            epochs (int): Number of epochs to train for.
        """
        if self.method == 1:
            method = 'Beta'
        else:
            method = 'Uniform'

        print('-------------------------------------------------------------------------------------------------------------------')
        print('| \t\t\t\t\t    MixUp Method: ', method)
        print('-------------------------------------------------------------------------------------------------------------------')
        print('|     Epoch \t | \t Train Loss \t | \t Train Acc \t | \t Test Loss \t | \t Test Acc ')
        print('-------------------------------------------------------------------------------------------------------------------')
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_model(train_loader)
            test_loss, test_acc = self.test_model(test_loader)
            train_loss, train_acc, test_loss, test_acc = map(lambda x: x.float() if not isinstance(x, float) else x, [train_loss, train_acc, test_loss, test_acc])
            print(f"| \t {epoch+1} \t | \t {train_loss:.4f} \t | \t {train_acc:.2f}% \t | \t {test_loss:.4f} \t | \t {test_acc:.2f}%")
            print('-------------------------------------------------------------------------------------------------------------------')

if __name__ == "__main__":
    
    train_loader, test_loader = load_cifar10_data(batch_size=36)
    images, targets = next(iter(test_loader))
    inv_normalize = transforms.Normalize(
                                        mean=[-m/s for m, s in zip((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))],
                                        std=[1/s for s in (0.2023, 0.1994, 0.2010)]
                                        )
    
    img_unnormalized = images[:36] 
    
    for i in range(len(img_unnormalized)):
        img_unnormalized[i] = inv_normalize(img_unnormalized[i])
    
    montage_mixup_png(img_unnormalized, 'test')
    

    print('\n')
    print('\t\t\t\t ########################################')
    print(' \t\t\t\t\t      MixUp Report')
    print('\t\t\t\t ########################################\n')
    
    for method in [1, 2]:
        mixup = MixUp(method=method, optimizer='sgd')
        mixup.train(train_loader=train_loader, test_loader=test_loader)         
        # Save model in same folder as file
        torch.save(mixup.resnet50.state_dict(), f"{os.path.dirname(os.path.abspath(__file__))}/model_method{method}.pt")
        montage_mixup_png(img_unnormalized, 'test')
        # get one sample from test loader and create montage and print predictions
        outputs = mixup.resnet50.forward(images)
        _, predictions = torch.max(outputs.data, 1)
        img_unnormalized = images[:36] 

        print('Test Classification')
        print('---------------------------')
        print("| Target | Predicted")
        print('---------------------------')

        for i in range(36):
            print("| {} \t | {}".format(targets[i].item(), predictions[i].item()))
        print('---------------------------\n')

    # get one sample from training loader and create montage
    images, _  = next(iter(train_loader) )
    images = images[:16]
    images = mixup.mix_up(images, targets)[0]
    for i in range(len(images)):
        images[i] = inv_normalize(images[i])
    
    montage_mixup_png(images, 'train')

  

    



