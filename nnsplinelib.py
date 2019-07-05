import numpy as np
from nnlib import *

class Spline1D(object):

    def __init__(self, max_points, optimizer=SGD(), epsilon=0.1): #x,y for initialization
        assert max_points >= 2
        self.n_points = max_points # changes dynamically
        self.n_max = max_points # max point is constant
        self.eps = epsilon
        
        X_ = np.random.uniform(-1+epsilon, 1-epsilon, size=(max_points-2))
        X = np.empty(shape=(self.n_points))
        X[1:-1] = X_
        X[0], X[-1] = -1-epsilon, 1+epsilon
        self.X = X
        self.Y = np.random.uniform(-1, 1, size=(max_points))
        
        self.XOpt = optimizer.set_parameter(self.X)
        self.YOpt = optimizer.set_parameter(self.Y)
        
        self.rangeX = None
        self.rangeX_n = None
        self.diffX = None
        self.diffY = None

        self.input = None
        self.output = None

        self.del_output = None
        self.del_X = None
        self.del_Y = None
        self.del_input=None

        self._sort_parameters_()
        pass

 ####################################################################################   
 ####################################################################################

    def _inrange_(self, X, break0, break1): #if x is after

        xmsk1 = X >= break0
        xmsk2 = X < break1
        xmsk = np.bitwise_and(xmsk1, xmsk2)
        xs = xmsk #*X
        return xs

    def _sort_parameters_(self,):
        sortindx = np.argsort(self.X)
        self.X = self.X[sortindx]
        self.Y = self.Y[sortindx]

    def _calculate_rangeX_(self,):

        rangeX = np.zeros((self.n_points-1, self.input.shape[0]))

        for i in range(self.n_points-1):

            if self.n_points-2 == 0:
                rangeX[i] = self._inrange_(self.input, -np.inf, np.inf)
            elif i==0:
                rangeX[i] = self._inrange_(self.input, -np.inf, self.X[i+1])
            elif i== self.n_points-2:
                rangeX[i] = self._inrange_(self.input, self.X[i], np.inf)
            else:
                rangeX[i] = self._inrange_(self.input, self.X[i], self.X[i+1])
        self.rangeX = rangeX

        rnx_ = np.count_nonzero(rangeX, axis=1)
        rangeX_n = np.zeros(self.n_points)
        rangeX_n[:-1] += rnx_
        rangeX_n[1:] += rnx_
        rangeX_n[rangeX_n == 0.] = -1.

        self.rangeX_n = rangeX_n

    def preprocess(self,):
        self.diffX = np.diff(self.X)
        self.diffY = np.diff(self.Y)

        # when diff between two X values is zero .: causes divided by zero error
        if (self.diffX == 0).any():
        # if len(self.diffX.nonzero()[0]) < len(self.diffX): 
            self._remove_close_points_()
            # self._sort_parameters_()


        self._sort_parameters_()
        self._calculate_rangeX_()


    def forward(self,input):
        self.input = input


        self.preprocess()

        output = np.zeros_like(self.input)
        for i in range(self.n_points-1):
            Y_ = self.diffY[i]/self.diffX[i] *(self.input - self.X[i]) + self.Y[i]
            output = output + Y_*self.rangeX[i]
        self.output = output
        return self.output

 ##################################################################################   

    def _backward_Y_(self,):

        consts = np.zeros((self.n_points-1, self.input.shape[0]))
        for i in range(self.n_points-1):
            consts[i] = (self.input-self.X[i])/self.diffX[i]

        dY = np.zeros((self.n_points, self.input.shape[0]))
        
        dY[0] = (-1* consts[0] +1) *self.rangeX[0]
        dY[-1] = consts[-1] *self.rangeX[-1]

        for i in range(1, self.n_points-1):
                a = consts[i-1]*self.rangeX[i-1]
                b = (-1* consts[i] +1)*self.rangeX[i]
                dY[i] = a+b
        dY = dY*self.del_output
        ## can choose any two options for gradient of Y
        dY = dY.sum(axis=1)/self.rangeX_n
        
        # dY_= dY.sum(axis=1)/self.rangeX_n
        # dY = dY.mean(axis=1)
        # dY[0], dY[-1] = dY_[0], dY_[-1]

        self.del_Y = dY
        return self.del_Y

    def _backward_X_(self,):

        consts = np.zeros((self.n_points-1, self.input.shape[0]))
        for i in range(self.n_points-1):
            consts[i] = self.diffY[i]/(self.diffX[i]**2)
        
        dX = np.zeros((self.n_points, self.input.shape[0]))
        dX[0] = consts[0]*(self.input - self.X[1])*self.rangeX[0]
        dX[-1] = -1*consts[-1]*(self.input - self.X[-2])*self.rangeX[-1]

        for i in range(1, self.n_points-1):
                a = -1*consts[i-1]*(self.input - self.X[i-1])*self.rangeX[i-1]
                b = consts[i]*(self.input - self.X[i+1])*self.rangeX[i]
                dX[i] = a+b
        dX = dX*self.del_output
        ##########This is true delX#############
        #Not Implemented
        ########################################

        ## can choose any of below options for calculating gradient of X
        dX = dX.mean(axis=1)
        # dX = dX.sum(axis=1)/self.rangeX_n

        self.del_X = dX
        return self.del_X

    def _backward_input_(self,):

            dinp = np.zeros_like(self.input)
            for i in range(self.n_points-1):
                dinp = dinp + self.diffY[i]/self.diffX[i] *self.rangeX[i]
            
            dinp = dinp*self.del_output
            self.del_input = dinp
            return self.del_input

    def backward(self, del_output):
        self.del_output = del_output
        self._backward_Y_()
        self._backward_X_()
        self._backward_input_()
        return self.del_input

####################################################################################

    def update(self, learning_rate=0.1):
        gradients = self.XOpt.compute_gradient(self.del_X)
        self.X -= gradients
        gradients = self.YOpt.compute_gradient(self.del_Y)
        self.Y -= gradients

        self._sort_parameters_()

        min = self.input.min()
        max = self.input.max()
        # if np.abs(self.X[0] - min) > self.eps:
        if self.X[0] > min:
            # print('changing minimum start point')
            self.preprocess()
            X_ = min-self.eps
            Y_ = self.diffY[0]/self.diffX[0] *(X_ - self.X[0]) + self.Y[0]
            self.X[0] = X_
            self.Y[0] = Y_
        # if np.abs(self.X[-1] - max) > self.eps:
        if self.X[-1] < max:
            # print('changing maximum end point')
            self.preprocess()
            X_ = max+self.eps
            Y_ = self.diffY[-1]/self.diffX[-1] *(X_ - self.X[-2]) + self.Y[-2]
            self.X[-1] = X_
            self.Y[-1] = Y_

        # self.X[0] = self.input.min()-self.eps
        # self.X[-1] = self.input.max()+self.eps



 ##################################################################################   

    def _remove_close_points_(self, min_dist=1e-3):
        # removing ones which are very close to each other
        # requires sorted points first
        x_diff = np.ones_like(self.X)
        x_diff[1:] = np.diff(self.X)
        clipmask = np.abs(x_diff) > min_dist
        self.X = self.X[clipmask]
        self.Y = self.Y[clipmask]
        self.n_points = len(self.X)
    
    def _combine_linear_points_(self, min_area=1e-2):
        triangle = np.ones_like(self.X)
        for i in range(self.n_points-2):
            triangle[i+1] = 0.5*np.abs(
                (self.X[i] - self.X[i+2])*self.diffY[i]+self.diffX[i]*(self.Y[i+2] - self.Y[i]))
        mergemask = triangle > min_area
        self.X = self.X[mergemask]
        self.Y = self.Y[mergemask]
        self.n_points = len(self.X)

    def _combine_highly_nonlinear_points_(self,):
        pass

    def  _add_new_point_(self, min_error=1e-4):
        # adding units where the error > min_error
        if self.n_points < self.n_max:
            dYs = np.zeros((self.n_points-1, self.input.shape[0]))
            for i in range(self.n_points-1):
                dYs[i] = self.del_output * self.rangeX[i]
            dYerr = (dYs**2).mean(axis=1)
            index = np.argmax(dYerr)
            if dYerr[index] > min_error:
                newpx = (self.X[index] + self.X[index+1])/2.
                newpy = (self.Y[index] + self.Y[index+1])/2.
                # adding new interpolation points
                self.X = np.append(self.X, newpx)
                self.Y = np.append(self.Y, newpy)
                # sorting the points for plotting
                self.n_points = len(self.X)
                self._sort_parameters_()
    
    def _remove_no_input_points_(self,):
        #removing if points contain no input
        self.preprocess()
        nx = np.zeros_like(self.X)
        nx_ = np.count_nonzero(self.rangeX, axis=1)
        nx[:-1] += nx_
        nx[1:] += nx_

        nx0mask = nx!=0
        self.X = self.X[nx0mask]
        self.Y = self.Y[nx0mask]
        self.n_points = len(self.X)

    def _increase_pieces_(self, increase_by=1):
        self.n_max += increase_by



 ##################################################################################   
 ##################################################################################   

class SplineVectorLayer(Layer):

    def __init__(self, input_dim, max_points, optimizer=SGD(),epsilon=0.1):
        self.dimension = input_dim
        self.spline_list = [Spline1D(max_points, optimizer=Optimizer, epsilon=epsilon) for _ in range(input_dim)]
        self.input = None
        self.output = None
        self.del_output = None

        layerList.append(self)

    def forward(self, input):
        self.input = input
        self.output = np.empty_like(input)
        for i in range(self.dimension):
            inpi = input[:, i]
            outi = self.spline_list[i].forward(inpi)
            self.output[:, i] = outi
        return self.output

    def backward(self, del_output):
        self.del_output = del_output
        del_input = np.empty_like(del_output)
        for i in range(self.dimension):
            del_outi = del_output[:, i]
            del_inpi = self.spline_list[i].backward(del_outi)
            del_input[:, i] = del_inpi
        return del_input

    def update(self, learning_rate=0.1):
        for spline in self.spline_list:
            spline.update(learning_rate)

    def _increase_pieces_(self, increase_by=1):
        for spline in self.spline_list:
            spline._increase_pieces_(increase_by)

    def _maintain_good_spline_(self):
        for spline in self.spline_list:
            spline._remove_close_points_()
            spline._combine_linear_points_()
            spline._remove_no_input_points_()
            spline._add_new_point_()

 ##################################################################################   
 ##################################################################################   

class SplineMatrixLayer(Layer):

    def __init__(self, input_dim, output_dim, max_points, optimizer=SGD(), epsilon=0.1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spline_mat = np.empty((input_dim, output_dim), dtype=np.object)
        for i in range(self.input_dim):
            for j in range(self.output_dim):
                self.spline_mat[i,j] = Spline1D(max_points, optimizer=Optimizer, epsilon=epsilon)
        self.input = None
        self.output = None
        self.del_output = None
        layerList.append(self)

    def forward(self, input):
        self.input = input
        m = self.input.shape[0]
        self.output = np.zeros((m, self.output_dim))

        for i in range(self.input_dim):
            inpi = self.input[:, i]
            for j in range(self.output_dim):
                self.output[:, j] += self.spline_mat[i,j].forward(inpi)
        return self.output

    def backward(self, del_output):
        self.del_output = del_output
        del_input = np.zeros_like(self.input)
        
        for i in range(self.output_dim):
            del_outi = del_output[:, i]
            for j in range(self.input_dim):
                del_input[:, j] += self.spline_mat[j,i].backward(del_outi)
        return del_input

    def update(self, learning_rate=0.1):
        for i in range(self.input_dim):
            for j in range(self.output_dim):
                self.spline_mat[i,j].update(learning_rate)


    def _increase_pieces_(self, increase_by=1):
        for i in range(self.input_dim):
            for j in range(self.output_dim):
                self.spline_mat[i,j]._increase_pieces_(increase_by)

    def _maintain_good_spline_(self):
        for i in range(self.input_dim):
            for spline in self.spline_mat[i]:
                spline._remove_close_points_()
                spline._combine_linear_points_()
                spline._remove_no_input_points_()
                spline._add_new_point_()


 ##################################################################################   
 ##################################################################################   
