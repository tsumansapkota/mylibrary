import numpy as np

class SplineManifold1D(object):

    def __init__(self, dimension, max_points, initial_points=2, epsilon=0.1):
        assert initial_points >=2
        assert max_points >= initial_points

        self.dimension = dimension
        self.max_points = max_points
        self.n_points = initial_points
        self.eps = epsilon

        X_ = np.random.uniform(-1, 1, size=(self.n_points-2, self.dimension))
        X = np.ones(shape=(self.n_points, self.dimension))
        X[1:-1] = X_
        X[0], X[-1] = -1*X[0], 1*X[-1]
        self.X = X

        self.inputs = None
        # self.outputs = None
        
        self.projection = None
        self.errors = None
        self.gradients = None
        self.range_index = None

        self.del_X = None

        pass

    def _get_projection_(self, data, point1, point2):
        data_ = data - point1
        point2_ = point2 - point1
        proj_ = data_@point2_.T * point2_ / np.sum(point2_**2)
        proj = proj_ + point1
        return proj

    # def forward(self, inputs):
    #     self.inputs = inputs
    #     self.projection = np.zeros_like(self.inputs)
    #     self.gradients = np.zeros_like(self.inputs)
    #     self.range_index = np.zeros(self.inputs.shape[0], dtype=np.int)
    #     self.errors = np.ones(self.inputs.shape[0])*9999
    #     for index in range(len(self.X)-1):
    #         point1 = self.X[index:index+1]
    #         point2 = self.X[index+1:index+2]
    #         projection = self._get_projection_(self.inputs, point1, point2)
    #         self._projtemp = projection
    #         minv = np.minimum(point1, point2)
    #         maxv = np.maximum(point1, point2)
    #         projection = np.clip(projection, minv, maxv)
            
    #         gradients = projection - self.inputs
    #         errors = (gradients**2).mean(axis=1)
    #         mask = self.errors > errors
    #         self.projection[mask] = projection[mask]
    #         self.gradients[mask] = gradients[mask]
    #         self.range_index[mask] = index
    #         self.errors[mask] = errors[mask]
    #         pass
        
    #     return self.projection

    def forward(self, inputs):
        self.inputs = inputs
        self.projection = np.zeros_like(self.inputs)
        self.gradients = np.zeros_like(self.inputs)
        self.range_index = np.zeros(self.inputs.shape[0], dtype=np.int)
        self.errors = np.ones(self.inputs.shape[0])*9999
        for index in range(len(self.X)-1):
            point1 = self.X[index:index+1]
            point2 = self.X[index+1:index+2]
            projection = self._get_projection_(self.inputs, point1, point2)
            self._projtemp = projection
            scaler = self._get_gradient_scaler_(projection, point1, point2)

            ### selecting and clipping seems similar
            ## this is selecting based approach
            # if len(self.X) > 2:
            #     if index > 0: ## points below the point1
            #         mask = scaler[:,0]>1
            #         projection[mask] = point1
            #     if index < len(self.X)-2: ## points above the point2 
            #         mask = scaler[:,0]<0
            #         projection[mask] = point2

            ## this is clipping based approach
            minv = np.minimum(point1, point2)
            maxv = np.maximum(point1, point2)
            projection = np.clip(projection, minv, maxv)
            
            gradients = projection - self.inputs
            errors = (gradients**2).mean(axis=1)
            mask = self.errors > errors
            self.projection[mask] = projection[mask]
            self.gradients[mask] = gradients[mask]
            self.range_index[mask] = index
            self.errors[mask] = errors[mask]
            pass
        
        return self.projection

    def _get_gradient_scaler_(self, projection, point1, point2):
        scaler = (projection - point2)/(point1 - point2)
        return scaler

    def backward(self):
        self.del_X = np.zeros_like(self.X)
        m = len(self.inputs)

        ### For EDGE points
        index = 0
        mask = self.range_index == index
        scaler = self._get_gradient_scaler_(self.projection[mask], self.X[index:index+1], self.X[index+1:index+2])
        self.del_X[index] = np.sum(self.gradients[mask]*scaler, axis=0)/m

        index = len(self.X)-1
        mask = self.range_index == index - 1
        scaler = self._get_gradient_scaler_(self.projection[mask], self.X[index:index+1], self.X[index-1:index])
        self.del_X[index] = np.sum(self.gradients[mask]*scaler, axis=0)/m

        ## For INNER Points
        for index in range(1,len(self.X)-1):
            mask = self.range_index == index
            scaler = self._get_gradient_scaler_(self.projection[mask], self.X[index:index+1], self.X[index+1:index+2])
            self.del_X[index] += np.sum(self.gradients[mask]*scaler, axis=0)/(2*m)

            mask = self.range_index == index - 1
            scaler = self._get_gradient_scaler_(self.projection[mask], self.X[index:index+1], self.X[index-1:index])
            self.del_X[index] += np.sum(self.gradients[mask]*scaler, axis=0)/(2*m)

        return

    def update(self, learning_rate=0.01):
        # self._maintain_manifold_()
        self.X -= learning_rate*self.del_X

    def _maintain_manifold_(self):
        for index in range(1, len(self.X)-1):
            mask = self.range_index == index-1
            pm = self.projection[mask]
            p1 = pm[np.argmin(np.abs(pm-self.X[index])[:,0])]

            mask = self.range_index == index
            pm = self.projection[mask]
            p2 = pm[np.argmin(np.abs(pm-self.X[index])[:,0])]

            self.X[index] = (p1+p2)/2            


    def add_point(self):
        ## find range with largest gradient
        max_err_indx = None
        max_err = 0
        for index in range(len(self.X)-1):
            mask = self.range_index == index
            err = np.mean(self.errors[mask])
            if err > max_err:
                max_err = err
                max_err_indx = index
            pass
        self.X = np.insert(self.X, max_err_indx+1, (self.X[max_err_indx]+self.X[max_err_indx+1])/2, axis=0)


        