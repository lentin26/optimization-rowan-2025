import numpy as np

def steepest_gradient_descent(x0, tol=1e-7, stepsize=1e-3, max_iter=10000):
    """
    Implements the steepest gradient descent algorithm with backtracking line search.
    
    Parameters:
    x0 : numpy array
        Initial guess.
    tol : float, optional
        Tolerance for stopping criteria, default is 1e-7.
    a : float, optional
        Scalar constant for the Rosenbrock function, default is 100.
    max_iter : int, optional
        Maximum number of iterations, default is 10000.
    
    Returns:
    x : numpy array
        Optimized solution.
    """
    def rosenbrock_gradient(x):
        """
        Calculates the gradient of the Rosenbrock function at a given point x.

        Args:
            x (numpy.ndarray): A 1D numpy array representing the point at which to evaluate the gradient.

        Returns:
            numpy.ndarray: A 1D numpy array representing the gradient of the Rosenbrock function at x.
        """
        n = len(x)
        grad = np.zeros(n)

        grad[0] = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
        grad[1] = 200*(x[1] - x[0]**2)

        return grad

    def general_rosenbrock(x):
        x = np.asarray(x)
        func_val = (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
        return func_val, rosenbrock_gradient(x)

    objfun = lambda x: general_rosenbrock(x)
    
    iter_count = 0
    stop = False
    x_old = np.array(x0)
    
    f, gradf = objfun(x_old)
    normgrad = np.linalg.norm(gradf)
    
    if normgrad < tol:
        stop = True
    
    trace = {'iter': [], 'obj_fun': [], 'norm_grad': [], 'tol': []}
    while not stop and iter_count < max_iter:
        p = -gradf  # replace with Hessian?? -inv(Hess)*gradf or solve linear equations faster
        stepsize = 1e-3  # Fixed step size for simplicity
        
        x_new = x_old + stepsize * p
        
        f, gradf = objfun(x_new)
        normgrad = np.linalg.norm(gradf)
        
        current_tol = np.linalg.norm(x_new - x_old)
        if current_tol < tol or normgrad < tol:
            stop = True
        
        x_old = x_new
        iter_count += 1

        trace['iter'].append(iter_count)
        trace['obj_fun'].append(f)
        trace['norm_grad'].append(normgrad)
        trace['tol'].append(current_tol)
        #print(f'iter={iter_count}, obj fun={f:.12f}, gradient norm={normgrad:.12f}')
    
    return x_old, trace

if __name__ == '__main__':
    # Example usage:
    x0 = 2 * np.ones(2)  # Initial guess
    tol = 1e-7
    optimized_x = steepest_gradient_descent(x0, tol, max_iter=10000)
    print("Optimized solution:", optimized_x)
