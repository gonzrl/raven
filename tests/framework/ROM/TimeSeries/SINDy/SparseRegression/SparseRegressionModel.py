import numpy as np

def run(self, Input):
    x = Input['x']
    self.y = 2. + 0.2*x**3 + 0.5*np.sin(3 * x)
