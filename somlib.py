import numpy as np


class SOM2D(object):

    def __init__(self,input_dim:int, output_dims:[tuple, list], learning_rate=0.3, sigma=2., decay_rate=0.999):
        self.input_dim = input_dim
        assert len(output_dims) == 2
        self.output_dims = output_dims
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.sigma = sigma

        ## the parameter
        self.weights = None
        self.del_weights = None
        ## for distance function
        self.distances = np.zeros(self.output_dims)
        self.differences = np.zeros((self.input_dim,*self.output_dims))
        self.min_dist_ij = None
        ## for neighbourhood influence
        self.factors = np.zeros(self.output_dims)
        self.count = 1.


        self._initialize_vector_()
        pass
    
    def _initialize_vector_(self):
        # self.weights = np.random.uniform(low=-1, high=1, size=(self.input_dim,*self.output_dims))
        self.weights = np.random.randn(self.input_dim,*self.output_dims)
    
    def _calculate_vector_distance_(self,input):
        self.input = input
        for i in range(self.output_dims[0]):
            for j in range(self.output_dims[1]):
                diffij = self.input-self.weights[:,i,j]
                distij = np.linalg.norm(diffij)
                self.differences[:, i,j] = diffij
                self.distances[i,j] = distij

        self.min_dist_ij = np.unravel_index(np.argmin(self.distances), self.distances.shape)
        return self.distances, self.min_dist_ij


    def _calculate_neighbourhood_influence_(self):
        for i in range(self.output_dims[0]):
            for j in range(self.output_dims[1]):
                fij = np.exp(
                    -(np.sqrt((self.min_dist_ij[0]-i)**2+(self.min_dist_ij[1]-j)**2)
                    /(2*self.sigma**2))
                    )
                self.factors[i,j] = fij
        return self.factors

    def _calculate_gradients_(self):
        self.del_weights = self.factors*self.differences

    def _update_(self):
        self.weights = self.weights + self.learning_rate*self.del_weights

    def _decay_(self):
        # decayer = np.exp(-self.count*self.decay_rate)
        decayer = 1/(1+self.count*self.decay_rate)
        self.learning_rate = self.learning_rate*decayer
        self.sigma = self.sigma*decayer
        self.count+=1
        