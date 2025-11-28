# solver.py
import numpy as np
import pandas as pd

class DuffingSolver:
    def __init__(self, params):
        """
        params: dict, 包含 delta, alpha, beta, gamma, omega
        """
        self.p = params

    def acceleration(self, x, v, t):
        return (self.p['gamma'] * np.cos(self.p['omega'] * t) 
                - self.p['delta'] * v 
                - self.p['alpha'] * x 
                - self.p['beta'] * x**3)

    def solve(self, t_max, dt, x0=0.0, v0=0.0):
        t_eval = np.arange(0, t_max, dt)
        n_steps = len(t_eval)
        
        x = np.zeros(n_steps)
        v = np.zeros(n_steps)
        x[0], v[0] = x0, v0
        
        a_curr = self.acceleration(x[0], v[0], t_eval[0])
        
        for i in range(n_steps - 1):
            t = t_eval[i]
            t_next = t_eval[i+1]
            
            # Velocity Verlet
            x[i+1] = x[i] + v[i]*dt + 0.5*a_curr*dt**2
            v_pred = v[i] + a_curr*dt
            a_next = self.acceleration(x[i+1], v_pred, t_next)
            v[i+1] = v[i] + 0.5*(a_curr + a_next)*dt
            a_curr = a_next
            
        return pd.DataFrame({'time': t_eval, 'x': x, 'v': v})
