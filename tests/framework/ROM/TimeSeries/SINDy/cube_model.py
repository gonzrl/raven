import numpy as np

def run(raven, _):
  raven.y = np.squeeze(getattr(raven, 'x', 0))**3
