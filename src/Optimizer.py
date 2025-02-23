import numpy as np

class Optimizer():
    def __init__(self, x0, a=1, tol=1e-7, stepsize=1e-3, max_iter=10000):
        self.x0 = x0
        self.tol = tol
        self.stepsize = stepsize
        self.max_iter = max_iter

    def fit(self, method, func):
        """
        Choose and execute an optimization method.
        """
        if func == 'rosenbrock':
            objfun = self.get_rosenbrock_obj_func()
        elif func == 'quadratic':
            objfun = self.quadratic_function()

        if method == 'steepest_grad_descent':
            return self.steepest_gradient_descent(objfun)
        elif method == 'newtons_method':
            return self.newtons_methods(objfun)
        else:
            raise \
                Exception('method` must be one on [steepest_grad_descent, newtons_methods]')

    def get_quadratic_obj_func(self, return_hess=True):

        def gradient(x, a):
            n = len(x)
            grad = np.zeros(n)
            grad[0] = 2*x[0]
            grad[1] = a*x[1]**2
            return grad

        def eval_func(x, a):
            return x[0]**2 + a*x[1]**2

        if not return_hess:
            objfun = lambda x, a: [eval_func(x, a), gradient(x, a)]
            return objfun
        else:
            objfun = lambda x, a: [eval_func(x, a), gradient(x, a)]
            return objfun

    def get_rosenbrock_obj_func(self):
        """
        Return objective function for Rosenbrock.
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
        return objfun

    def newtons_methods(self, objfun):
        iter_count = 0
        stop = False
        x_old = np.array(self.x0)
        
        f, gradf = objfun(x_old)
        normgrad = np.linalg.norm(gradf)
        
        if normgrad < self.tol:
            stop = True
        
        trace = {'iter': [], 'obj_fun': [], 'norm_grad': [], 'tol': []}
        while not stop and iter_count < self.max_iter:
            p = -inv(Hess)*gradf 
            stepsize = 1e-3  # Fixed step size for simplicity
            
            x_new = x_old + stepsize * p
            
            f, gradf = objfun(x_new)
            normgrad = np.linalg.norm(gradf)
            
            current_tol = np.linalg.norm(x_new - x_old)
            if current_tol < self.tol or normgrad < self.tol:
                stop = True
            
            x_old = x_new
            iter_count += 1

            trace['iter'].append(iter_count)
            trace['obj_fun'].append(f)
            trace['norm_grad'].append(normgrad)
            trace['tol'].append(current_tol)
        
        return x_old, trace

    def steepest_gradient_descent(self, objfun):
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
        iter_count = 0
        stop = False
        x_old = np.array(self.x0)
        
        f, gradf = objfun(x_old)
        normgrad = np.linalg.norm(gradf)
        
        if normgrad < self.tol:
            stop = True
        
        trace = {'iter': [], 'obj_fun': [], 'norm_grad': [], 'tol': []}
        while not stop and iter_count < self.max_iter:
            p = -gradf  # replace with Hessian?? -inv(Hess)*gradf or solve linear equations faster
            stepsize = 1e-3  # Fixed step size for simplicity
            
            x_new = x_old + stepsize * p
            
            f, gradf = objfun(x_new)
            normgrad = np.linalg.norm(gradf)
            
            current_tol = np.linalg.norm(x_new - x_old)
            if current_tol < self.tol or normgrad < self.tol:
                stop = True
            
            x_old = x_new
            iter_count += 1

            trace['iter'].append(iter_count)
            trace['obj_fun'].append(f)
            trace['norm_grad'].append(normgrad)
            trace['tol'].append(current_tol)
        
        return x_old, trace

