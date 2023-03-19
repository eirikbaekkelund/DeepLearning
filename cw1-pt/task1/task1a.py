import torch
from task import generate_data, polynomial_func

def mse_loss(x, t, w):
    return torch.mean(torch.square((polynomial_func(x, w) - t)))

def fit_polynomial_sgd_M(x, t, M_max, lr, batch_size, print_loss=False):
    """ 
    Runs a stochastic gradient descent for fitting polynomial functions with the 
    optimal degree. Note that M here is the length of the weight vector w,
    However, the degree of the polynomial function is M-1.

    Args:
        x (torch.Tensor): Input data.
        t (torch.Tensor): Target values.
        M_max (int): Highest degree the polynomial function can have.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        print_loss (bool): If True, the function prints the loss at each epoch.
    
    Returns:
        M (int): Optimal degree of the polynomial function.
        w (torch.Tensor): Optimal weight vector.
        loss (float): Loss at the end of the training.
    """
    w = torch.randn(M_max+1, requires_grad=True)
    M = torch.randint(1, M_max, size=(1,), dtype=torch.float32, requires_grad=True)
   
    optimizer = torch.optim.SGD((w, M), lr=lr, momentum=0.9)

    n_batches = len(x) // batch_size
    print_freq = 10

    best_loss = torch.inf
    mask = torch.arange(start=0, end=M_max+1, dtype=torch.float32)

    print_freq = 10
   
    
    for epoch in range(100):
        permutation = torch.randperm(len(x))
        for i in range(n_batches):
            
            indices = permutation[i*batch_size:(i+1)*batch_size]
            
            t_batch = t[indices]
            x_batch = x[indices]
            
            weights_mask = torch.relu(M - mask).unsqueeze(-1)
            weights_mask = torch.clamp(weights_mask, 0, 1)

            w_new =  weights_mask.t() * w
            w_new = w_new.reshape(-1,)
            
            optimizer.zero_grad()
            loss = mse_loss(x_batch, t_batch, w_new)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(w, max_norm=1)
            torch.nn.utils.clip_grad_norm_(M, max_norm=1)
            optimizer.step()
            
            if loss < best_loss:
                best_loss = loss
                best_w = w_new
                best_M = M.int() - 1 if torch.abs(M.int() - M) < 0.5 else M.int()
            
        if epoch % print_freq == 0 and print_loss:
            M_print = M.int() - 1 if torch.abs(M.int() - M) < 0.5 else M.int()
            print(f"| {epoch} \t | {M_print.item()} \t | {loss.item():.2f}")
            print('--------------------------------------------------')

    
    return best_M, best_w

if __name__ == "__main__":
    
    x_train, t_train, x_test, t_test = generate_data(100)
    
    print('\n')
    print('--------------------------------------------------')
    print('|\t TASK 1A (OPTIMAL LEADING DEGREE SGD)')
    print('--------------------------------------------------')
    print('| Epoch  | M \t | Batch Loss ')
    print('--------------------------------------------------')
    
    M, w = fit_polynomial_sgd_M(x=x_train, 
                                t=t_train, 
                                M_max=12, 
                                lr=0.01, 
                                batch_size=5, 
                                print_loss=True)
    loss = mse_loss(x_test, t_test, w)

    print('\n')

    print('--------------------------------------------------------------------------------------------------')
    print('| \t\t\t\t BEST PARAMETERS (MAX DEGREE M = 12)')
    print('--------------------------------------------------------------------------------------------------')
    print('| Predicted M \t | ', M.item())
    print('--------------------------------------------------------------------------------------------------')
    print('| True M \t | ', 4)
    print('--------------------------------------------------------------------------------------------------')
    print(f'| Predicted w \t |  { [ round(weight, 2) for weight in w.detach().numpy().tolist()  ] }')
    print('--------------------------------------------------------------------------------------------------')
    print(f'| True w \t |  { [i for i in range(1,6)]}')
    print('--------------------------------------------------------------------------------------------------')
    print(f'| Test Loss \t | {loss.item():.2f}')
    print('--------------------------------------------------------------------------------------------------\n\n')

    
    print(' \t\t\t\t Predicted Curve vs. True Curve')
    print('--------------------------------------------------------------------------------------------------')
    print('| Std Dev Test \t | \t Mean Test \t | \t Std Dev Train \t | \t Mean Train \t ')
    print('--------------------------------------------------------------------------------------------------')
    print(f'| {torch.std(polynomial_func(x_test, w) - t_test).item():.2f} \t | \t {torch.mean(polynomial_func(x_test, w) - t_test):.2f} \t | \t {torch.std(polynomial_func(x_train, w) - t_train).item():.2f} \t | \t {torch.mean(polynomial_func(x_train, w) - t_train).item():.2f} ')
    print('--------------------------------------------------------------------------------------------------')

