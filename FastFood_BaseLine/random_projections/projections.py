import numpy as np
from sklearn.base import TransformerMixin

class rks(TransformerMixin): 
    def __init__(self, new_dim, scale):
        self.n_proj = new_dim
        self.scale = scale
        self.random_state = np.random.randint(100) 

    #get random weight matrice for projection
    def fit(self,x):
        #random seed
        rng = np.random.RandomState(self.random_state)
        #random gaussian for mult
        self.rand_gaussian = (1 / self.scale) * rng.normal(size=(x.shape[1], self.n_proj))
        #random offset
        self.rand_offset = rng.uniform(0, 2 * np.pi, size=self.n_proj)
        return self

    #applies transform
    def transform(self, x):
        #mult gaussian
        mult = x @ self.rand_gaussian + self.rand_offset
        #nonlinearity
        return np.sqrt(2 / self.n_proj) * np.cos(mult)


class fastfood():
    def __init__(self, new_dim, scale):
        self.new_dim = new_dim
        self.scale = scale
        self.random_state = np.random.randint(100)
    
    def fit(self, x):
        #input features
        d = x.shape[0]
        #random state
        rng = np.random.RandomState(self.random_state)

        self.B = np.diag(rng.choice([-1, 1], size=d))
        self.perm = rng.permutation(d)
        self.G = np.diag(rng.normal(d))

