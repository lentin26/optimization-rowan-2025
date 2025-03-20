

class Algorithm5_1:

    def __init__(self):
        self.x = None

    def fit(self, A, b, x=0):
        """
        Solve Ax = b. 
        params:
            x: initial guess.
        """
        r = A @ x - b
        p = -r
        k = 0
        while r.all() != 0:
            a = - (r.T @ p) / (p.T @ A @ p)
            x = x + a * p
            r = A @ x - b
            β = (r.T @ A @ p) / (p.T @ A @ p)
            p = -r + β * p
            k += 1

        # update solution
        return x, k


