import numpy as np

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
    def __init__(self, indices, spline_nd):
        '''
        The points must list of indices
        '''
        self.indices = indices
        self.spline_nd = spline_nd

    def get_indices(self):
        return self.indices

    def get_points(self):
        return self.spline_nd.X[self.indices], self.spline_nd.Y[self.indices]

    def is_point_inside(self, point):
        if point is not np.ndarray:
            point = np.array(point)

        ones = np.ones([len(self.indices),1])
        R_ = self.spline_nd.X[self.indices]
        # Ri = np.zeros_like(R)

        R = np.hstack([R_, ones])
        detR = np.linalg.det(R)
        # Ri = np.hstack([Ri, ones])
        # print(R, detR)
        for i, ind in enumerate(self.indices):
            Rtemp = R.copy()
            Rtemp[i,:-1] = point
            # print('...\n', Rtemp)
            # Ri = Ri + Rtemp
            # print(Ri)
            det = np.linalg.det(Rtemp)
            # print(det)
            if det*detR < 0: return False
        return True

    

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
        self.root:Region = None

        self.input = None
        self.output = None

        self._initialize_()


    def _initialize_(self,):
        self.X = np.random.uniform(low=-1, high=1, size=[self.input_dim+1, self.input_dim])
        self.Y = np.zeros(self.input_dim+1)

        self.root = Region([i for i in range(self.input_dim+1)], self)
        return

    def forward(self, input):
        self.input = input
        # print(self.X)
        # print(self.Y)
        # print(self.root.get_points())
        # print('________')
        # print(self.root.is_point_inside(input))
        isInside = self.root.is_point_inside(input)
        return isInside
