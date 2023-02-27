# import warnings
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=UserWarning)
import torch
import time
import random

def polynomial_func(x, w=torch.tensor([1,2,3,4,5,0],dtype=torch.float32)):
    """
    Implements a polynomoial function that takes two input arguments, a weight vector w of size M + 1 
    and a scalar x. The function returns the value of the polynomial function at x. The polynomial function is vectorized for multiple pairs of scalar x and weight vector w.

    Args:
        w (torch array): Parameters of the polynomial function.
        x (scalar): Input data.

    Returns:
        np.ndarray: Output data.

    """

    x_powers = torch.pow(x.unsqueeze(1), torch.arange(len(w))) # [x^0, x^1, ..., x^M]
    x_powers = x_powers.reshape(x_powers.shape[0], -1) # Reshape x_powers to shape (n, M+1)
    return torch.matmul(x_powers, w.t()) # w[0]x^0 + w[1]x^2 + ... + w[M]x^M + w[M+1]x^(M+1)


def generate_data(sample_size, x_range=[-20,20], std_dev=0.2): 
    """ 
    Generates data in the range of x_range. The data is generated by adding Gaussian noise to the output of the polynomial function.
    The observed values are obtained by adding Gaussian noise to the output of the polynomial function.

    Args:
        sample_size (int): Number of data points.
        x_range (list): Range of x values.
        std_dev (float): Standard deviation of the Gaussian noise.
    
    Returns:
        x_train (torch.Tensor): Input data for training.
        t_train (torch.Tensor): Target values for training.
        x_test (torch.Tensor): Input data for testing.
        t_test (torch.Tensor): Target values for testing.
    """
    x = torch.linspace(x_range[0], x_range[1], 100 + sample_size) # [x_1, x_2, ..., x_n]
    t = polynomial_func(x) # [y_1, y_2, ..., y_n]
    t += torch.normal(0, std_dev, t.shape) # [y_1 + noise_1, y_2 + noise_2, ..., y_n + noise_n]
    
    idx = list(range(sample_size + 100))
    random.shuffle(idx)

    t_train = t[idx[:sample_size]]
    t_test = t[idx[sample_size:]]
    x_train = x[idx[:sample_size]]
    x_test = x[idx[sample_size:]]

    return x_train.squeeze(), t_train.squeeze(), x_test.squeeze(), t_test.squeeze()

def fit_polynomial_ls(x, t, M):
    """ 
    Using linear least squares method to fit a polynomial function to the data.
    Takes M pairs of x and t and returns the optimal weight vector w.

    Args:
        x (torch.Tensor): Input data.
        t (torch.Tensor): Target values.
        M (int): Degree of the polynomial function.
    
    Returns:
        w (torch.Tensor): Optimal weight vector.
    """
    powers = torch.arange(M+1).to(x.dtype) # [0, 1, ..., M]
    x_powers = torch.pow(x.unsqueeze(1), powers) # [x^0, x^1, ..., x^M]
    w = torch.matmul(torch.matmul(torch.inverse( torch.matmul(x_powers.t(), x_powers)) , x_powers.t()), t)
    
    return w


def fit_polynomial_sgd(x, t, M, lr, batch_size, print_loss=False):
    """ 
    Runs a stochastic gradient descent for fitting polynomial functions with the 
    same arguments as fit_polynomial_ls in addition to the learning rate and the batch size.

    Args:
        x (torch.Tensor): Input data.
        t (torch.Tensor): Target values.
        M (int): Degree of the polynomial function.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        print_loss (bool): If True, the function prints the loss at each epoch.
    
    Returns:
        w (torch.Tensor): Optimal weight vector.
        
        If loss is printed:
        w (torch.Tensor): Optimal weight vector.
        losses (list): List of losses at each epoch.
        epochs (list): List of epochs at which the loss is printed. 
    """
    w = torch.randn(M+1, requires_grad=True)
    optimizer = torch.optim.SGD([w], lr=lr)

    n_batches = len(x) // batch_size
    losses = []
    epochs = []
    print_freq = 10
    
    for epoch in range(400):
        permutation = torch.randperm(len(x))
        for i in range(n_batches):
            indices = permutation[i*batch_size:(i+1)*batch_size]
            t_batch = t.rename(None)[indices]
            x_batch = x[indices]
            optimizer.zero_grad()
            loss = torch.mean(torch.square((polynomial_func(x_batch, w) - t_batch)))
            loss.backward()
            
            optimizer.step()

        losses.append(loss.item())

        if epoch % print_freq == 0 and print_loss:
            epochs.append(epoch)
            losses.append(loss.item())
            print(f"Epoch: {epoch},\t | \t Loss: {loss.item()}")
    
    if print_loss:
        return w, losses, epochs
    else:
        return w

def print_header():
    print('\n')
    print("##################################    TASK 1    ################################\n")
    print('\n')

def print_weights_and_errors(sample_sizes=[50,100]):
    """ 
    Reports the mean prediction error for the training and test sets for both the least squares and the minibatch SGD methods.
    The mean prediction error is computed as the mean squared error between the target values and the predicted values.

    Args:
        sample_sizes (list): List of sample sizes.
    """
    for sample in sample_sizes:

        x_train, t_train, x_test, t_test = generate_data(sample)
        w_ls_train = fit_polynomial_ls(x_train, t_train, 5)
        w_sgd_train = fit_polynomial_sgd(x_train, t_train, 5, 2.377e-13, 10)
        w_ls_test = fit_polynomial_ls(x_test, t_test, 5)
        w_sgd_test = fit_polynomial_sgd(x_test, t_test, 5, 2.377e-13, 10)

        weights = [0,1,2,3,4,5]
        print(f"\t\t\t TRAINING SET | SAMPLE SIZE = {sample}")
        print("--------------------------------------------------------------------------------")
        print("\t\t\t\t|  Least Squares |      Minibatch SGD ")
        print('--------------------------------------------------------------------------------')
        print(f'| Mean Prediction Error \t|\t {round(torch.mean(torch.abs(polynomial_func(x_test, w_ls_train) - t_test)).detach().numpy().tolist(),3)} \t | \t {round(torch.mean(torch.abs(polynomial_func(x_test, w_sgd_train) - t_test)).detach().numpy().tolist(),3)}')
        print(f'| Standard Dev Prediction Error |\t {round(torch.std(torch.abs(polynomial_func(x_test, w_ls_train) - t_test)).detach().numpy().tolist(),3)} \t | \t {round(torch.std(torch.abs(polynomial_func(x_test, w_sgd_train) - t_test)).detach().numpy().tolist(),3)}')
        print('--------------------------------------------------------------------------------')

        print('\n')
        print(f"\t\t\t TRAINING WEIGHTS | SAMPLE SIZE {sample}")
        print('--------------------------------------------------------------------------------')
        print("|      Weight \t |     Least Squares Weights  \t |   Minibatch SGD Weights ")
        print('--------------------------------------------------------------------------------')
        
        for i in range(6):
            print(f"|\t {weights[i]} \t | \t\t {w_ls_train[i]:.2f} \t\t | \t       {w_sgd_train[i]:.2f}")
        
        print('--------------------------------------------------------------------------------')
        print('\n')
        print('\n')
    
        print(f"\t\t\t TEST SET | SAMPLE SIZE = {sample}")
        print("--------------------------------------------------------------------------------")
        print("\t\t\t\t|  Least Squares |      Minibatch SGD ")
        print('--------------------------------------------------------------------------------')
        print(f'| Mean Prediction Error \t| \t {round(torch.mean(torch.abs(polynomial_func(x_test, w_ls_test) - t_test)).detach().numpy().tolist(),3)} \t | \t {round(torch.mean(torch.abs(polynomial_func(x_test, w_sgd_test) - t_test)).detach().numpy().tolist(),3)}')
        print(f'| Standard Dev Prediction Error | \t {round(torch.std(torch.abs(polynomial_func(x_test, w_ls_test) - t_test)).detach().numpy().tolist(),3)} \t | \t {round(torch.std(torch.abs(polynomial_func(x_test, w_sgd_test) - t_test)).detach().numpy().tolist(),3)}')
        print('--------------------------------------------------------------------------------')

        print('\n')
        print(f"\t\t\t TEST WEIGHTS | SAMPLE SIZE {sample}")
        print('--------------------------------------------------------------------------------')
        print("|      Weight \t |     Least Squares Weights  \t |   Minibatch SGD Weights ")
        print('--------------------------------------------------------------------------------')
        
        for i in range(6):
            print(f"|\t {weights[i]} \t | \t\t {w_ls_test[i]:.2f} \t\t | \t      {w_sgd_test[i]:.2f}")
        
        print('--------------------------------------------------------------------------------')
        print('\n')
        print('\n')

def print_test_results(sample_sizes=[50,100]):
    """ 
    Reports the RMSE of the function values for the test set for both the least squares and the minibatch SGD methods.

    Args:
        sample_sizes (list): List of sample sizes.
    """
    weights_actual = torch.cat((torch.tensor(torch.arange(1, 6)), torch.tensor([0])))
    for sample in sample_sizes:
        _, _, x_test, t_test = generate_data(sample)
        
        w_ls_test = fit_polynomial_ls(x_test, t_test, 5)
        w_sgd_test = fit_polynomial_sgd(x_test, t_test, 5, 2.377e-13, 10, print_loss=False)

        print(f"\t\t\t PREDICTED RESULTS | SAMPLE SIZE = {sample}")
        print("--------------------------------------------------------------------------------")
        print("\t\t\t\t |  Least Squares  |     Minibatch SGD ")
        print('--------------------------------------------------------------------------------')
        print(f'| RMSE Function Values \t\t |     {round( torch.sqrt(torch.mean((polynomial_func(x_test, w_ls_test) - t_test) ** 2)).detach().numpy().tolist(),3)} \t   | \t {round(torch.sqrt(torch.mean((polynomial_func(x_test, w_sgd_test) - t_test) ** 2)).detach().numpy().tolist(),3)}')
        print(f'| RMSE Weights \t\t\t |     {round( torch.sqrt(torch.mean((w_ls_test - weights_actual) ** 2)).detach().numpy().tolist(),3)} \t   | \t {round(torch.sqrt(torch.mean((w_sgd_test - weights_actual) ** 2)).detach().numpy().tolist(),3)}')
        print('--------------------------------------------------------------------------------')
        print('\n')

def print_speed_comparison(M=5, sample_size = [50,100]):
    """ 
    Compare the speed of the least squares method and the stochastic gradient descent method.

    Args:
        M (int): The degree of the polynomial.
        sample_size (list): List of sample sizes.
    """
    
    times_ls = []
    times_sgd = []

    for sample in sample_size:
        x_train, t_train, _, _ = generate_data(sample)
        start = time.time()
        fit_polynomial_ls(x_train, t_train, M)
        end = time.time()
        times_ls.append(end - start)

        start = time.time()
        fit_polynomial_sgd(x_train, t_train, M, lr=2.377e-13, batch_size=10)
        end = time.time()
        times_sgd.append(end - start)

    print(f"\t\t\t\t TIMED RESULTS") 
    print("--------------------------------------------------------------------------------")
    print("\t\t |  Least Squares \t | \t Minibatch SGD \t | SAMPLE SIZE")
    print('--------------------------------------------------------------------------------')
    print(f'| Times  \t | \t {round(times_ls[0],4)} [s] \t | \t {round(times_sgd[0],3)} [s]\t | \t {sample_size[0]}')
    print(f'| Times  \t | \t {round(times_ls[1],4)} [s] \t | \t {round(times_sgd[1],3)} [s]\t | \t {sample_size[1]}')
    print('--------------------------------------------------------------------------------')
    print('\n')

if __name__ == '__main__':
    print_header()
    print_weights_and_errors()
    print_test_results()
    print_speed_comparison()