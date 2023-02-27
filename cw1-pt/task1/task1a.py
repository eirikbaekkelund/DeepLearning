import torch
from task import generate_data, polynomial_func
import matplotlib.pyplot as plt

def ridge_loss(y, t, w, lmbda):
    """ 
    Ridge loss to emphasize the importance of the weights and powers of the polynomial.
    
    Args:
        y (torch.Tensor): Predicted values.
        t (torch.Tensor): Target values.
        w (torch.Tensor): Weights of the polynomial.
        lmbda (float): Regularization parameter.
    Returns:
        torch.Tensor: Ridge loss.
    """
    return torch.mean(torch.square((y - t))) + lmbda * torch.norm(w, p=2)

def fit_polynomial_sgd_M(x, t, M, lr, lmbda=0.1, batch_size=5):
    """ 
     Minibatch Stochastic Gradient Descent to fit a polynomial of degree M to the data.
    Args:
        x (torch.Tensor): Input data.
        t (torch.Tensor): Input labels.
        M (int): Degree of the polynomial.
        lr (float): Learning rate.
        lmbda (float): Regularization parameter.
        batch_size (int): Batch size.
    Returns:
        torch.Tensor: Weights of the polynomial.
    """
    # Initialize parameters
    w = torch.randn(M+1, requires_grad=True)
    
    optimizer = torch.optim.SGD([w], lr=lr)   
    best_loss = torch.inf    
    n_batches = len(x) // batch_size

    for _ in range(200):
        permutation = torch.randperm(len(x))
        for i in range(n_batches):
            indices = permutation[i*batch_size:(i+1)*batch_size]
            t_batch = t[indices]
            x_batch = x[indices]

            y = polynomial_func(x_batch, w)
            # using regularization to put importance on neccessary weights and powers
            loss = ridge_loss(y, t_batch, w, lmbda)
            if loss < best_loss:
                best_loss = loss
                best_w = w
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
       
     
    return best_w, best_loss

def print_learned_M(M_range=torch.arange(0,6), lrs=[1e-2, 1e-3, 1e-5, 1e-8, 1e-10, 1e-13], lmbdas = [100], batch_size=20):
    """
    Prints the report of the weights of the polynomial for different degrees M.

    Args:
        M_range (torch.Tensor): Range of degrees M.
        lrs (list): List of learning rates.
        lmbdas (list): List of regularization parameters.
        batch_size (int): Batch size.
    """
    x_train, t_train, x_test, t_test = generate_data(100)
    print('\n\n')
    print(f"\t\t LEARNING DEGREE M | Batch Size: {batch_size} | RIDGE REGRESSION IN SGD")
    print("-----------------------------------------------------------------------------------------------------------")
    print("| M | Lambda \t | Learning Rate \t | Loss \t\t | \t Weights Vector")
    print('-----------------------------------------------------------------------------------------------------------')
    
    best_loss_M = torch.inf    
    
    for M, lr in zip(M_range, lrs):
        
        best_loss = torch.inf
        best_lmbda = torch.inf
        best_lr = torch.inf
   
        for lmbda in lmbdas:
            w, _ = fit_polynomial_sgd_M(x_train, t_train, M, lr, lmbda, batch_size)
            loss = ridge_loss(polynomial_func(x_test, w), t_test, w, lmbda)
            if loss < best_loss:
                best_loss = loss
                best_lmbda = lmbda
                best_lr = lr
    
        w, _ = fit_polynomial_sgd_M(x_train, t_train, M, best_lr, best_lmbda, batch_size)
        loss = ridge_loss(polynomial_func(x_test, w), t_test, w, best_lmbda)
        
        if best_loss < best_loss_M:
            best_loss_M = best_loss
            best_learned_M = M
            best_weights_M = w

        if M == 4:

            print(f"| {M} | {best_lmbda} \t | {best_lr} \t\t | {round(loss.detach().numpy().tolist(),2)} \t\t |  {[round(weight, 2) for weight in w.detach().numpy().tolist()]}")
        else:
            print(f"| {M} | {best_lmbda} \t | {best_lr} \t\t | {round(loss.detach().numpy().tolist(),2)} \t |  {[round(weight, 2) for weight in w.detach().numpy().tolist()]}")

    print("-----------------------------------------------------------------------------------------------------------")
    print(f"| \t BEST M = {best_learned_M} \t | \t LOSS = {best_loss_M}")
    print("-----------------------------------------------------------------------------------------------------------")
    print('\n')
    print('\n')
    
    plt.title('Test vs Prediction Scatter Plot')
    plt.scatter(x_train, t_train, marker = 'o', label='true values')
    plt.scatter(x_train, polynomial_func(x_train, best_weights_M).detach(), marker = 'x', label='predicted values')
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    print_learned_M(batch_size=20)