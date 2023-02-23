import torch
import numpy as np
import math
import time

def generate_data(w, sample_size, x_range=[-20,20], std_dev=0.2): 
    """ 
    Generates data in the range of x_range. The data is generated by adding Gaussian noise to the output of the polynomial function.
    The observed values are obtained by adding Gaussian noise to the output of the polynomial function.
    """
    x = torch.linspace(x_range[0], x_range[1], sample_size) # [x_1, x_2, ..., x_n]
    y = polynomial_func(x, w) # [y_1, y_2, ..., y_n]
    y += torch.normal(0, std_dev, y.shape) # [y_1 + noise_1, y_2 + noise_2, ..., y_n + noise_n]
    return x, y

def polynomial_func(x, w):
    """Implements a polynomoial function that takes two input arguments, a weight vector w of size M + 1 
    and a scalar x. The function returns the value of the polynomial function at x. The polynomial function is vectorized for multiple pairs of scalar x and weight vector w.

    Args:
        w (torch array): Parameters of the polynomial function.
        x (scalar): Input data.

    Returns:
        np.ndarray: Output data.

    """
    powers = torch.arange(len(w)).to(x.dtype)  # [0, 1, ..., M]
    x_powers = torch.pow(x.unsqueeze(1), powers) # [x^0, x^1, ..., x^M] 
   
    return torch.matmul(x_powers, w) # w[0]x^1 + w[1]x^2 + ... + w[M]x^M

def fit_polynomial_ls(x, t, M):
    """ 
    Using linear least squares method to fit a polynomial function to the data.
    Takes M pairs of x and t and returns the optimal weight vector w.
    """
    powers = torch.arange(M).to(x.dtype) # [0, 1, ..., M]
    x_powers = torch.pow(x.unsqueeze(1), powers) # [x^0, x^1, ..., x^M]
    w = torch.matmul(torch.matmul(torch.inverse( torch.matmul(x_powers.t(), x_powers)) , x_powers.t()), t)
    
    return w

def fit_polynomial_sgd(x, t, M, lr, batch_size):
    """ 
    Runs a stochastic gradient descent for fitting polynomial functions with the 
    same arguments as fit_polynomial_ls in addition to the learning rate and the batch size.
    """
    w = torch.randn(M, requires_grad=True)
    #w = torch.tensor(torch.randn(M).clone().detach(), requires_grad=True) 
    optimizer = torch.optim.SGD([w], lr=lr)

    n_batches = len(x) // batch_size
    losses = []
    for _ in range(1000):
        permutation = torch.randperm(len(x))
        for i in range(n_batches):
            indices = permutation[i*batch_size:(i+1)*batch_size]
            t_batch = t[indices]
            x_batch = x[indices]
            optimizer.zero_grad()
            loss = torch.mean(torch.square(polynomial_func(x_batch, w) - t_batch))
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
    
    return w

def print_report(M, w=torch.tensor([1,2,3,4,5], dtype=torch.float32), lr=2.377e-10, batch_size=None,sample_size = [50,100], method='sgd'):
    """ 
    Print the mean and the standard deviation in difference between the observed training data
    and the underlying true polynomial function for the stochastic gradient descent/LS method.
    It also prints the difference between the predicted values and the underlying true polynomial.
    """
    assert w.shape[0] == M, "The number of weights should be equal to the degree of the polynomial + 1"
    assert method in 'ls' or 'sgd', "Method should be either \'ls\' or \'sgd\'"
    
    if method == 'sgd':
        print('Method: SGD')
        print("Hyperparameter Settings:\n")
        print("Learning rate:", lr)
        print("Batch size:", batch_size)
        print("--------------------------------------------------------------------------------\n")
    else:
        print('Method: LS')
        print("--------------------------------------------------------------------------------\n")
    
    for sample in sample_size:
        x, t = generate_data(w, sample)
        print("Sample size:", sample)
        print("--------------------------------------------------------------------------------")
        print(f"Target range (t): [{round(torch.min(t).numpy().tolist(),3)}, {round(torch.max(t).numpy().tolist(),3)}]")
        print(f"mean: {round(torch.mean(t).numpy().tolist(),3)}, std: {round(torch.std(t).numpy().tolist(),3)}")
        print('--------------------------------------------------------------------------------')

        if method == 'ls':
            w_pred = fit_polynomial_ls(x, t, M)
        else:
            w_pred = fit_polynomial_sgd(x, t, M, lr, batch_size)

        print("True weights: \t\t\t\t\t", w.numpy().tolist())
        print("Predicted weights: \t\t\t\t", [round(elem, 3) for elem in w_pred.detach().numpy().tolist()])
        print("Difference between true and predicted weights: ", [round(elem, 3) for elem in (w - w_pred).detach().numpy().tolist()])

        print("Mean difference between true and predicted weights: \t\t\t", round(torch.mean(w - w_pred).detach().numpy().tolist(), 4))
        print("Standard deviation of difference between true and predicted weights: \t", round(torch.std(w - w_pred).detach().numpy().tolist(),4))
        print("Difference between true and predicted values: \t\t\t\t", round(torch.mean(torch.abs(polynomial_func(x, w) - polynomial_func(x, w_pred))).detach().numpy().tolist(),4) )
        print("Standard deviation of difference between true and predicted values: \t", round(torch.std(torch.abs(polynomial_func(x, w) - polynomial_func(x, w_pred))).detach().numpy().tolist(),4))
        print("--------------------------------------------------------------------------------\n")

def compare_speed(M, w=torch.tensor([1,2,3,4,5], dtype=torch.float32), sample_size = [50,100]):
    """ 
    Compare the speed of the least squares method and the stochastic gradient descent method.
    """
    print("Comparison of speed between the least squares method and the stochastic gradient descent method:\n")
    print("--------------------------------------------------------------------------------\n")
    for sample in sample_size:
        x, t = generate_data(w, sample)
        print("Sample size:", sample)
        print("--------------------------------------------------------------------------------")

        start = time.time()
        w_ls = fit_polynomial_ls(x, t, M)
        end = time.time()
        print("Least Squares Method: ", end - start, "seconds")

        start = time.time()
        w_sgd = fit_polynomial_sgd(x, t, M, lr=2.377e-10, batch_size=10)
        end = time.time()
        print("Stochastic Gradient Descent Method: ", end - start, "seconds")
        print("--------------------------------------------------------------------------------\n")

def learn_M(M_range, lrs, batch_size):
    """ 
    Returns the M that minimizes the loss for the stochastic gradient descent method.
    """
    losses = []
    best_loss = np.inf
    best_M = np.inf
    best_lr = np.inf
    for M, lr in zip(M_range,lrs):
        x, t = generate_data(torch.tensor([i for i in range(1, M+1)], dtype=torch.float32), 100)
        w = fit_polynomial_sgd(x, t, M, lr, batch_size)
        loss = torch.mean(torch.square(polynomial_func(x, w) - t))
        losses.append(loss.item())
        if loss < best_loss:
            best_loss = loss
            best_M = M
            best_lr = lr

    return best_M, best_lr, best_loss

def print_learnead_M(M_range, lrs, batch_size):
    """ 
    Print the M that minimizes the loss for the stochastic gradient descent method.
    """
    print("M that minimizes the loss for the stochastic gradient descent method:\n")
    print("--------------------------------------------------------------------------------\n")
    print("Learning rates:", lrs)
    print("Batch size:", batch_size)
    print("--------------------------------------------------------------------------------\n")
    print("M:", learn_M(M_range, lrs, batch_size))
    print("--------------------------------------------------------------------------------\n")

def report_M(M_range, lrs, batch_size, w=torch.tensor([1,2,3,4,5], dtype=torch.float32), sample_size = [50,100]):
    """ 
    Print the mean and the standard deviation in difference between the observed training data
    and the underlying true polynomial function for the stochastic gradient descent method.
    It also prints the difference between the predicted sgd values and the underlying true polynomial.
    """
    print("--------------------------------------------------------------------------------\n")
    print("M that minimizes the loss for the stochastic gradient descent method:\n")
    print("--------------------------------------------------------------------------------\n")
    print("Learning rates:", lrs)
    print("Batch size:", batch_size)
    print("--------------------------------------------------------------------------------\n")
    for sample in sample_size:
        x, t = generate_data(w, sample)
        print("Sample size:", sample)
        print("--------------------------------------------------------------------------------")
        print(f"Target range (t): [{round(torch.min(t).numpy().tolist(),3)}, {round(torch.max(t).numpy().tolist(),3)}]")
        print(f"mean: {round(torch.mean(t).numpy().tolist(),3)}, std: {round(torch.std(t).numpy().tolist(),3)}")
        print('--------------------------------------------------------------------------------')

        M, lr, loss = learn_M(M_range, lrs, batch_size)
        print("M:", M)
        print("Learning rate:", lr)
        print("Loss:", loss)
        print("--------------------------------------------------------------------------------\n")

if __name__ == '__main__':
    print_report(M=5, lr=2.377e-10, batch_size=10, method='sgd')
    print_report(M=5, method='ls')
    compare_speed(M=5)
    print_learnead_M(M_range=[i for i in range(1, 11)], lrs= [2e-2, 1e-3, 2e-5, 2e-8, 2e-10, 2e-18, 2e-22, 2e-24, 2e-26, 2e-28], batch_size = 10)
    report_M(M_range=[i for i in range(1, 11)], lrs=[2e-2, 1e-3, 2e-5, 2e-8, 2e-10, 2e-18, 2e-22, 2e-24, 2e-26, 2e-28], batch_size = 10)