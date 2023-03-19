from classes import MixUp
from utils import load_cifar10_train_val_test
import torch
from torch.autograd import Variable
import time
import os

class Ablation(MixUp):
    """
    Class for comparing adam vs. sgd 
    Includes a validation tester as compared with task2.
    """
    def __init__(self, optimizer, num_classes):
        super().__init__(optimizer=optimizer)
        self.num_classes = num_classes
  
    def f1_score(self, targets, true_positives, false_positives, false_negatives):
        """
        Calculates the F1 score for a batch of predictions.
        """
        epsilon = 1e-7
        precision = true_positives / (true_positives + false_positives + epsilon)
        recall = true_positives / (true_positives + false_negatives + epsilon)
        f1_scores = 2 * precision * recall / (precision + recall + epsilon)

        weights = torch.zeros(self.num_classes)
        for i in range(self.num_classes):
            weights[i] = (targets == i).sum().item()

        weights /= weights.sum()
        val_f1 = (f1_scores * weights).sum().item()
        return val_f1

        

    def val_model(self, val_loader):
        """ 
        Tests the model on the validation data.
        
        Args:
            model (torchvision model): Model to test.
            val_loader (torch.utils.data.DataLoader): Validation data loader.
        Returns:
            val_loss (float): Validation loss.
            val_acc (float): Validation accuracy.
            val_f1 (float): Validation F1 score.
        """
        val_loss = 0
        correct = 0
        total = 0
  
        true_positives = torch.zeros(self.num_classes)
        false_positives = torch.zeros(self.num_classes)
        false_negatives = torch.zeros(self.num_classes)
        
        self.resnet50.eval()
        
        with torch.no_grad():
            for data, targets in val_loader:

                inputs, targets = Variable(data), Variable(targets)
                outputs = self.resnet50(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                # Update true positives, false positives, and false negatives for F1 score calculation
                for i in range(self.num_classes):
                    true_positives[i] += ((predicted == i) & (targets == i)).sum().item()
                    false_positives[i] += ((predicted == i) & (targets != i)).sum().item()
                    false_negatives[i] += ((predicted != i) & (targets == i)).sum().item()

            val_f1 = self.f1_score(targets=targets,
                                   true_positives=true_positives, 
                                   false_positives=false_positives, 
                                   false_negatives=false_negatives)
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * (correct / total)
            
        return val_loss, val_acc.item(), val_f1
        
    def train(self, train_loader, val_loader, test_loader, epochs=10):
        """
        Trains and tests the model.

        Args:
            model (torchvision model): Model to train and test.
            train_loader (torch.utils.data.DataLoader): Training data loader with mixup.
            test_loader (torch.utils.data.DataLoader): Test data loader.
            epochs (int): Number of epochs to train for.
        Returns:
            train_losses (list): List of training losses.
            train_accs (list): List of training accuracies.
            test_losses (list): List of test losses.
            test_accs (list): List of test accuracies.
        """
        if isinstance(self.optimizer, torch.optim.SGD):
            mode = 'SGD'
        else:
            mode = 'Adam'
        print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------')
        print('| \t\t\t\t\t\t\t\t       Optimizer: ', mode)
        print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------')
        print(f'| Epoch \t | Train Loss \t | Train Acc \t | Val Loss \t | Val Acc \t | Val F1-Score \t | Test Loss \t | Test Acc \t | Time') 
        print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------')
        
        for epoch in range(epochs):
            start = time.time()
            train_loss, train_acc = self.train_model(train_loader)
            val_loss, val_acc, val_f1 = self.val_model(val_loader)
            test_loss, test_acc = self.test_model(test_loader)
            end = time.time()

            print(f'| \t {epoch+1} \t | {train_loss:.3f} \t | {train_acc:.3f}% \t |  {val_loss:.3f} \t | {val_acc:.3f}% \t | \t {val_f1:.3f} \t\t | {test_loss:.3f} \t | {test_acc:.3f}% \t | \t {(end-start) // 60} min(s)')
            print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------')

        print('\n')

if __name__ == "__main__":
    print('\n')
    train_loader, val_loader, test_loader = load_cifar10_train_val_test()
    print('\n')
    print('\t\t\t\t\t\t\t ########################################')
    print(' \t\t\t\t\t\t\t\t      Ablation Report')
    print('\t\t\t\t\t\t\t ########################################\n')

    ablation_sgd = Ablation(num_classes=10, optimizer='sgd')
    ablation_sgd.train(train_loader, val_loader, test_loader, epochs=10)

    ablation_adam = Ablation(num_classes=10, optimizer='adam')
    ablation_adam.train(train_loader, val_loader, test_loader, epochs=10)

    torch.save(ablation_sgd.resnet50.state_dict(), f"{os.path.dirname(os.path.abspath(__file__))}/model_ablation_sgd.pt")
    torch.save(ablation_adam.resnet50.state_dict(), f"{os.path.dirname(os.path.abspath(__file__))}/model_ablation_adam.pt")


