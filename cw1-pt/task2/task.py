import torch
import torch.nn as nn
import os
from utils import load_cifar10_data, montage_mixup_png
from classes import CustomResNet50

class Task(nn.Module):
    """ 
    Task that uses mixup and SGD for classification
    Using one of two sampling methods for mixup.
    They come implicity from the MixUp class applied to the training data.
    
    1. Beta distribution
    2. Uniform distribution
    """
    def __init__(self):
        super(Task, self).__init__()       
        self.resnet50 = CustomResNet50()
        self.criterion = nn.CrossEntropyLoss()
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
        y = y.long()
        y_inferred = y_inferred.long()
        
        return torch.mean(lam * self.criterion(pred, y) + (1 - lam) * self.criterion(pred, y_inferred))
    
    def train(self, train_loader, device, n_epochs=10):
        """ 
        Trains the model on the training data.
        
        Args:
            model (torchvision model): Model to train.
            train_loader (torch.utils.data.DataLoader): Training data loader with mixup.
            optimizer (torch.optim): Optimizer to use.
            device (torch.device): Device to use.
        Returns:
            torchvision model: Trained model.
        """
        print('==> Starting training...')
        
        self.resnet50.train()

        for epoch in range(n_epochs):
            for data, target, target_mix, lam in train_loader:
                x_mix, y, y_infer, lam = data.to(device), target.to(device), target_mix.to(device), lam.to(device)        

                pred = self.resnet50.forward(x_mix)
                loss = self.mixup_loss(pred, y, y_infer, lam)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            print('Epoch: {} \tLoss: {:.6f}'.format(epoch + 1, loss.item()))

        print('==> Finished training...')

    def test(self, test_loader, device, create_montage = True):
        """ 
        Tests the model on the test data.
        
        Args:
            model (torchvision model): Model to test.
            test_loader (torch.utils.data.DataLoader): Test data loader.
            device (torch.device): Device to use.
        """
        print('==> Testing model...\n')

        test_loss = 0
        correct = 0
        n_batches = int(len(test_loader.dataset) / test_loader.batch_size)
        self.resnet50.eval()

        with torch.no_grad():
            
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.resnet50.forward(data)
                test_loss += torch.sum(self.criterion(output, target.float())).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                target = target.argmax(dim=1, keepdim=True)
                correct += torch.mean((pred.squeeze() == target.squeeze()).float())
                
                if create_montage:
                    # undo normalization on data 
                    img_unnormalized = data[:36] * torch.Tensor((0.2023, 0.1994, 0.2010)).reshape(3, 1, 1)

                    montage_mixup_png(img_unnormalized, 'test')
                    
                    print('\t Test Classification')
                    print('------------------------------')
                    print("| Target \t | Predicted")
                    print('------------------------------')
                    
                    for i in range(36):
                        print("| {} \t | \t {}".format(target[i].item(), pred[i].item()))
                    
                    print('------------------------------\n')
                    
                    create_montage = False
        # number of batches
        n_batches = int(len(test_loader.dataset) / test_loader.batch_size)
        test_loss /= n_batches
        
        print('\t\t Test Report')
        print('-----------------------------------------------')
        print(f'Average loss | {test_loss}')
        print(f'Accuracy \t| {correct/ n_batches * 100:.2f} %')
        print('==> Finished testing...')

if __name__ == "__main__":
    print('\n')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)      
    print('\n')
    for method in [1, 2]:

        train_loader, test_loader = load_cifar10_data(sampling_method=method)
        resnet = Task().to(device)
        resnet.train(train_loader=train_loader, device=device) 
        print('\n')

        resnet.test(test_loader=test_loader, device=device)
        print('\n')
        
        # Save model in same folder as file
        torch.save(resnet.resnet50.state_dict(), f"{os.path.dirname(os.path.abspath(__file__))}/model_{method}.pt")

