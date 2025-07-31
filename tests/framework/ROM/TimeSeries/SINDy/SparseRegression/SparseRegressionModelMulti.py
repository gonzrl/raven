import numpy as np

def run(self, Input):
    x1 = Input['x1']
    x2 = Input['x2']
    self.y1 = 2. + 0.2*x1**3 + 0.5*np.sin(3 * x2)
    self.y2 = 4. + 0.1*x2**4 + 0.3*np.cos(2 * x1)
