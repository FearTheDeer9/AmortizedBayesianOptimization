# Mechanism factories
def constant_mechanism(value):
    pass

def linear_mechanism(weights, intercept=0.0):
    pass

def nonlinear_mechanism(func):
    pass

# Noise generators
def gaussian_noise(mean=0.0, std=1.0):
    pass

def uniform_noise(low=0.0, high=1.0):
    pass