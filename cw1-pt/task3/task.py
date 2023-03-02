from task2.task import ResNet, MixUp, load_cifar10_data
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

class Ablation(ResNet):
    def __init__(self, optimizer, sampling_method):
        super(Ablation).__init__()
        assert sampling_method in [1,2], "Sampling method must be either 1 or 2. 1 - Beta sampling, 2 - Uniform sampling (for mixup)"
        assert optimizer in ['sgd', 'adam'], "Optimizer must be either 'sgd' or 'adam'"
        
        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.resnet50.parameters(), lr=0.01, momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(self.resnet50.parameters(), lr=0.01)        
    
    def train_test_split(self, test_size=0.2):
        """ 
        Loads the cifar10 data and splits into train and test sets.
        
        Args:
            test_size (float): the proportion of the data to be used as the test set
        """
        self.train_loader, self.test_loader = load_cifar10_data(sampling_method=self.sampling_method, test_size=test_size)
    
    def random_split_validation(self, validation_size=0.1):
        """ 
        Splits the train set into train and validation sets.
        Uses a random subset of the train set as the validation set.

        Args:
            validation_size (float): the proportion of the train set to be used as the validation set
        """
        # getting random indices
        idx = np.random.randint(0, len(self.train_loader.dataset), int(len(self.train_loader.dataset)*validation_size))
        
        # creating validation and train sets
        self.validation_loader = DataLoader(Subset(self.train_loader.dataset, idx))
        self.train_loader = DataLoader(Subset(self.train_loader.dataset, np.setdiff1d(np.arange(len(self.train_loader.dataset)), idx)))
        
        # get batch size
        self.batch_size = self.train_loader.batch_size
    
    def train(self, device, epochs=10):
        """ 
        Trains the model for a given number of epochs.

        Args:
            epochs (int): the number of epochs to train the model for
        """
        self.train_test_split()
        self.random_split_validation()
        self.resnet50.train()

        print("==> Starting training...\n")
        
        print('\t\t\t\t Training Report')
        print("|----------------------------------------------------------------------------------------------------------|")
        print(f"| Epoch \t | Train Loss \t | Train Accuracy \t | Validation Loss \t | Validation Accuracy")
        print("|----------------------------------------------------------------------------------------------------------|")
        

        # TODO add another metric during training other than loss and accuracy
        
        for i in range(epochs):
            
            train_loss = 0
            n_correct_train = 0
            
            for data, target, target_mix, lam in self.train_loader:
                data, target, target_mix, lam = data.to(device), target.to(device), target_mix.to(device), lam.to(device)
                
                self.optimizer.zero_grad()
                
                output = self.resnet50(data)
                loss = self.mixup_loss(output, target, target_mix, lam)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                
                pred_train = output.argmax(dim=1, keepdim=True)
                target_train = target.argmax(dim=1, keepdim=True)
                
                n_correct_train += torch.mean((pred_train == target_train).float()).item()
            
            val_loss = 0
            n_correct_val = 0
            
            for data_val, target_val, target_mix_val, lam_val in self.validation_loader:
                data_val, target_val, target_mix_val, lam_val = data_val.to(device), target_val.to(device), target_mix_val.to(device), lam_val.to(device)
                
                output_val = self.resnet50(data_val)
                loss_val = self.mixup_loss(output_val, target_val, target_mix_val, lam_val)
                
                val_loss += loss_val.item()

                pred_val = output_val.argmax(dim=1, keepdim=True)
                target_val = target_val.argmax(dim=1, keepdim=True)

                n_correct_val += torch.mean((pred_val == target_val).float()).item()

            train_loss = (train_loss / self.batch_size ).item()
            validation_loss = (val_loss / self.batch_size ).item()
            
            mean_correct_train = (n_correct_train / self.batch_size).item()
            mean_correct_val = (n_correct_val / self.batch_size).item()
            
            print(f"| {i+1} \t | {train_loss:.4f} \t\t | {mean_correct_train:.4f} \t\t | {validation_loss:.4f} \t\t | {mean_correct_val:.4f}")
        
        print("|---------------------------------------------------------------------------------------------------------- \n")

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ablation = Ablation()

    ablation_sgd = Ablation('sgd', sampling_method=1)
    ablation_sgd.train(device=device, epochs=10)
    ablation.test(ablation_sgd.test_loader, device=device, create_montage=False)

    ablation_adam = Ablation('adam', sampling_method=1)
    ablation_adam.train(device=device, epochs=10)
    ablation.test(ablation_adam.test_loader, device=device, create_montage=False)
