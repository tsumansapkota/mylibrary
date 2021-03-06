import numpy as np
import abc
from itertools import combinations 

'''
- We need to divide the whole input space into regions.
- Each region consists of N+1 composing points.
- The regions are made inside regions in a tree structure.
- Each Parent Node has N+1 Child Node; eg: 2D input needs triangle, each new point on triangle gives 3 regions

- eg: FOR 2D input:

                    P1 P2 P3
                    /   |   \ 
                  /     |     \ 
           P1,P2,Q   P2,P3,Q   P1,P3,Q

- The points can be represented by global list of numbers index.

'''
# class Region():
#     def __init__(self, dimensions, points):
#         '''
#         The points must be NDarray of shape (dim+1, dim)
#         '''
#         self.dimensions = dimensions
#         self.points = points


# class Node():
#     def __init__(self, input_dim:int):
#         self.input_dim:int = input_dim
#         self.region:Region = None

#     def _initialize_root_region_(self, points=None):
#         if points is None:
#             points = np.random.uniform(low=-1, high=1.0, size=(self.input_dim+1, self.input_dim))
#         self.region = Region(self.input_dim, points)
#         return



class Region():
    def __init__(self, indices, SND):
        '''
        The points must list of indices
        '''
        self.indices = indices
        self.SND = SND
        assert len(self.indices) == self.SND.input_dim + 1

        self.coeff = None
        self.inside_indx = None

        self.temp_input = None

    def get_indices(self):
        return self.indices

    def get_points(self):
        return self.SND.X[self.indices], self.SND.Y[self.indices]

    def is_point_inside(self, point):
        if point is not np.ndarray:
            point = np.array(point)
        ##### The algorithm defined here at : http://steve.hollasch.net/cgindex/geometry/ptintet.html
        ones = np.ones([len(self.indices),1])
        R_ = self.SND.X[self.indices]

        R = np.hstack([R_, ones])
        detR = np.linalg.det(R)
        for i in range(len(self.indices)):
            Rtemp = R.copy()
            Rtemp[i,:-1] = point
            det = np.linalg.det(Rtemp)
            
            if det*detR < 0:
                return False
        return True

    def get_points_inside(self, points):
        inside = []
        for i, pt in enumerate(points):
            if self.is_point_inside(pt):
                inside.append(i)
        return np.array(inside, dtype=np.int)

    def calculate_interpolation_coefficient_matrix(self):
        X_temp,Y_ = self.get_points()

        ones = np.ones([self.SND.input_dim+1, 1])
        X_ = np.hstack([X_temp, ones])
        self.X_inv = np.linalg.inv(X_)
        self.coeff = self.X_inv @ Y_

        return self.coeff

    def forward(self):
        self.inside_indx = self.get_points_inside(self.SND.input)
        if len(self.inside_indx) == 0: return

        self.calculate_interpolation_coefficient_matrix()

        inputs = self.SND.input[self.inside_indx]
        inputs_new = np.hstack([inputs, np.ones([len(inputs), 1])])
        self.temp_input = inputs_new
        outputs = inputs_new @ self.coeff

        self.SND.output[self.inside_indx] = outputs
        return #outputs

    def backward(self):
        if len(self.inside_indx) == 0: return

        del_output = self.SND.del_output[self.inside_indx]
        m = len(self.inside_indx)
        # countig the number of gradients
        self.SND.count[self.indices] = self.SND.count[self.indices] + m

        del_coeff = (self.temp_input.T @ del_output )#/m 
        # print((self.SND.input[self.inside_indx]).T.shape, del_output.shape)

        # adding all the gradients
        Ygrad = self.X_inv.T @ del_coeff
        self.SND.del_Y[self.indices] = self.SND.del_Y[self.indices] + Ygrad
        # Xgrad = ((Ygrad/m)@self.coeff.T)[:,:-1]
        # Xgrad = (Ygrad@self.coeff.T)[:,:-1]
        Xgrad = Ygrad@((self.coeff[:-1]).T)
        self.SND.del_X[self.indices] = self.SND.del_X[self.indices] + Xgrad        

        # return (del_output @ self.coeff[:-1].T)
        self.SND.del_input[self.inside_indx] = del_output @ self.coeff[:-1].T
        return
 
class Node():

    def __init__(self, indices, SND):
        self.indices = indices
        self.SND = SND
        assert len(self.indices) == self.SND.input_dim + 1

        self.child = Region(indices, SND)
        self.splitNodes = []
        self.is_branch = False
        self.split_index = None
        #self.child = [R1, R2, ... ]

    def forward(self):
        if self.is_branch: ### if the Node is decision node and branches
            for node in self.splitNodes:
                node.forward()
        else:
            self.child.forward()
    
    def backward(self):
        if self.is_branch: ### if the Node is decision node and branches
            for node in self.splitNodes:
                node.backward()
        else:
            self.child.backward()

    def break_region(self):
        if not self.is_branch:
            # OX, OY = self.child.get_points()
            # newX, newY = OX.mean(axis=0, keepdims=True), OY.mean(axis=0, keepdims=True)
            newX = (self.SND.input[self.child.inside_indx]).mean(axis=0, keepdims=True)
            newY = (self.SND.Y[self.indices]).mean(axis=0, keepdims=True)

            self.split_index = len(self.SND.X)

            self.SND.X = np.append(self.SND.X, newX, axis=0)
            self.SND.Y = np.append(self.SND.Y, newY, axis=0)

            self.SND._init_gradients_()

            comb_points = list(combinations(self.indices, len(self.indices)-1) )
            # print(comb_points)
            self.splitNodes = []
            self.is_branch = True
            for cpt in comb_points:
                cpt = list(cpt); cpt.append(self.split_index)
                # print(cpt)
                newNode = Node(cpt, self.SND)
                self.splitNodes.append(newNode)
        else:
            OX, OY = self.child.get_points()
            newX, newY = OX.mean(axis=0, keepdims=True), OY.mean(axis=0, keepdims=True)

            self.SND.X[self.split_index] = newX
            self.SND.Y[self.split_index] = newY

            self.SND._init_gradients_()

            comb_points = list(combinations(self.indices, len(self.indices)-1) )
            # print(comb_points)
            self.splitNodes = []
            for cpt in comb_points:
                cpt = list(cpt); cpt.append(self.split_index)
                # print(cpt)
                newNode = Node(cpt, self.SND)
                self.splitNodes.append(newNode)

    def get_maximum_error_node(self):
        if self.is_branch: ### if the Node is decision node and branches
            max_err = 0.
            me_node = None
            for node in self.splitNodes:
                error, enode = node.get_maximum_error_node()
                # print('---', error)
                if error > max_err:
                    max_err, me_node = error, enode
            # print(max_err)
            return max_err, me_node

        else:
            del_output = self.SND.del_output[self.child.indices]
            err = (del_output**2).mean()
            # err = (del_output**2).sum()
            # err = np.abs(del_output).mean()
            # err = np.abs(del_output).sum()



            return err, self
    


        
    


class SplineND(object):

    def __init__(self, input_dim:int, max_points:int=None, num_points:int=None, epsilon:float=0.1):

        #### Hyper Parameters
        self.input_dim = input_dim
        self.max_points = max_points if max_points is not None else input_dim+1
        assert self.max_points > self.input_dim ## for 1d, 2pts ; 2d, 3pts ; ...
        self.epsilon = epsilon
        self.num_points = num_points if num_points is not None else max_points

        self.X = None
        self.Y = None
        self.count = None
        self.root:Node = None #Node([i for i in range(self.input_dim+1)], self)

        self.input = None
        self.inside_indx = None
        self.output = None

        self.del_output = None
        self.del_X = None
        self.del_Y = None

        self._initialize_()


    def _init_gradients_(self):
        self.del_X = np.zeros_like(self.X)
        self.del_Y = np.zeros_like(self.Y)
        self.count = np.zeros_like(self.Y, dtype=np.int)

        

    def _initialize_(self,):
        # self.X = np.random.uniform(low=-1, high=1, size=[self.input_dim+1, self.input_dim])
        self.X = np.random.normal(0, 1, size=[self.input_dim+1, self.input_dim])
        # self.Y = np.zeros((self.input_dim+1, 1))
        self.Y = np.random.uniform(size=(self.input_dim+1, 1))
        self._init_gradients_()

        self.root = Node([i for i in range(self.input_dim+1)], self)

        return

    def make_root_global_coverage(self, globalX):
        '''
        globlalX should have shape (n, nD...)
        '''
        for gx in globalX:
            if not self.root.child.is_point_inside(gx):

                # pts = self.root.get_points()[0]
                indices = self.root.indices
                pts = self.X[indices]

                center_pt = np.mean(pts, axis=0,keepdims=True)
                cpts = pts - center_pt

                direction = cpts/np.linalg.norm(cpts, ord=2, axis=1, keepdims=True)
                # print(direction)

                diffs = pts - gx
                dists = np.linalg.norm(diffs, ord=2, axis=1) + 1e-5 ### for gradient in the update section
                indx = np.argmin(dists)
                
                gindx = indices[indx]
                self.X[gindx] = pts[indx] + direction[indx]*dists[indx]  ### update
        return
    
    def make_root_global_coverage2(self, globalX=None):
        if globalX is None:
            globalX = self.input

        for _ in range(10):
            BREAK_NOW = True
            for gx in globalX:
                if not self.root.child.is_point_inside(gx):
                    BREAK_NOW = False
                    indices = self.root.indices
                    pts = self.X[indices]

                    center_pt = np.mean(pts, axis=0,keepdims=True)
                    cpts = pts - center_pt

                    direction = cpts/np.linalg.norm(cpts, ord=2, axis=1, keepdims=True)
                    # print(direction)

                    diffs = pts - gx
                    dists = np.linalg.norm(diffs, ord=2, axis=1) + 1e-5 ### for gradient in the update section
                    indx = np.argmin(dists)
                    
                    gindx = indices[indx]
                    self.X[gindx] = pts[indx] + direction[indx]*dists[indx]*0.1  ### update
            if BREAK_NOW: break

        return

    def add_new_point(self):
        err, node = self.root.get_maximum_error_node()
        node.break_region()
        return        

    def forward(self, input):
        self.input = input
        self.output = np.zeros([len(input), 1])

        # self.output[self.inside_indx] = self.root.forward(self.input[self.inside_indx])
        self.root.forward()

        # isInside = self.root.is_point_inside(input)
        ########### interpolate y given x for all points ############
        

        return self.output

    def backward(self, del_output):
        self.del_output = del_output
        self.del_input = np.zeros_like(self.input)

        self.root.backward()
        return self.del_input

    def update(self, lr=0.1):
        ######## Method 1    
        # self.count = self.count + 1
        # gradX = (self.del_X/self.count)
        # gradY = (self.del_Y/self.count)
        # self.X = self.X - lr*gradX
        # self.Y = self.Y - lr*gradY

        ######## Method 2
        # self.X = self.X - lr*self.del_X
        # self.Y = self.Y - lr*self.del_Y


        ######## Method 3
        m = len(self.input)    
        gradX = self.del_X/m
        gradY = self.del_Y/m
        # gradY = self.del_Y/(self.count+1)
        #####################################

        self.X = self.X - lr*gradX
        self.Y = self.Y - lr*gradY

        self.count *= 0
        self.del_X *= 0
        self.del_Y *= 0


