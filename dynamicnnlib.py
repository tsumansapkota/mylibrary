import numpy as np

try:
    import nnlib as tnn
except ImportError:
    import mylibrary.nnlib as tnn
    pass


class Relu_1Linear(tnn.Layer):
    def __init__(self):
        tnn.layerList.append(self)
        self.x = None
        self.mask = None
        self.del_output = None


    def forward(self, x):
        self.x = x
        self.mask = ((self.x >= 0).astype(float))
        self.mask[:, 0] = self.mask[:, 0]*0 + 1.
        return self.x*self.mask

    def backward(self, output_delta):
        self.del_output = output_delta
        return self.mask * output_delta    


class DynamicNN_Relu:

    def __init__(self, layers_dim = [], optimizer=tnn.SGD(), ):
        assert len(layers_dim) > 1

        self.layers_dim = layers_dim
        self.optimizer = optimizer

        ### Model
        self.layers = []
        self.relus = []
        self.residuals_0  = []
        self.residuals_1 = []

        self.residuals_dim = []

        self._initialize_layers_()

    def _initialize_layers_(self):
        for i in range(len(self.layers_dim)-1):
            layer = tnn.LinearLayer(self.layers_dim[i],
                                    self.layers_dim[i+1],
                                    optimizer=self.optimizer)
            res0 = tnn.NonLinearLayer(self.layers_dim[i],
                                      1,
                                      activation=Relu_1Linear(),
                                      optimizer=self.optimizer)
            res1 = tnn.LinearLayer(1,
                                   self.layers_dim[i+1],
                                   optimizer=self.optimizer)
            res1.weights *= 0.
            relu = Relu_1Linear()
            if i == len(self.layers_dim)-2:
                relu = tnn.Linear()
            self.layers.append(layer)
            self.relus.append(relu)
            self.residuals_0.append(res0)
            self.residuals_1.append(res1)
            self.residuals_dim.append(1)

    ############# Basic Network Operations
    
    def forward(self, input):
        for i in range(len(self.layers)):
            out0 = self.layers[i].forward(input)
            h0 = self.residuals_0[i].forward(input)
            h1 = self.residuals_1[i].forward(h0)
            output = self.relus[i].forward(out0+h1)
            input = output
        return output

    def backward(self, del_output):
        for i in reversed(range(len(self.layers))):
            del_output = self.relus[i].backward(del_output)
            del_input0 = self.layers[i].backward(del_output)
            del_h0 = self.residuals_1[i].backward(del_output)
            del_input1 = self.residuals_0[i].backward(del_h0)
            del_input = del_input0 + del_input1
            del_output = del_input
        return del_input

    def update(self):
        for i in range(len(self.layers)):
            self.layers[i].update()
            self.residuals_0[i].update()
            self.residuals_1[i].update()

    ############# Basic Network Operations
    ############# Network Increase Operation

    def add_neurons_to_all_possible_layers(self, increase_by=1):
        l_ = []
        r0_ = []
        r1_ = []

        new_layers_dim = self.layers_dim.copy()
        new_residuals_dim = self.residuals_dim.copy()
        for i in range(1, len(self.layers_dim)-1 ):
            new_layers_dim[i] += increase_by
            new_residuals_dim[i] += increase_by
        new_residuals_dim[0] += increase_by    

        for i in range(len(self.layers)):
            layer = tnn.LinearLayer(new_layers_dim[i],
                                    new_layers_dim[i+1],
                                    optimizer=self.optimizer)
            res0 = tnn.NonLinearLayer(new_layers_dim[i],
                                    new_residuals_dim[i],
                                    activation=Relu_1Linear(),
                                    optimizer=self.optimizer)
            res1 = tnn.LinearLayer(new_residuals_dim[i],
                                new_layers_dim[i+1],
                                optimizer=self.optimizer)
            ### copying and zeroing weights
            layer.weights[:self.layers_dim[i], :self.layers_dim[i+1]] = self.layers[i].weights
            layer.bias[:self.layers_dim[i+1]] = self.layers[i].bias
            # the outgoing weights will be zero if the incoming dimension has changed
            layer.weights[self.layers_dim[i]:new_layers_dim[i]] *= 0.
            # similar for residual layer
            res0.weights[:self.layers_dim[i], :self.residuals_dim[i]] = self.residuals_0[i].weights
            res0.bias[:self.residuals_dim[i]] = self.residuals_0[i].bias
            res0.weights[self.layers_dim[i]:new_layers_dim[i]] *= 0.
            # for another residual layer
            res1.weights[:self.residuals_dim[i], :self.layers_dim[i+1]] = self.residuals_1[i].weights
            res1.bias[:self.layers_dim[i+1]] = self.residuals_1[i].bias
            res1.weights[self.residuals_dim[i]:new_residuals_dim[i]] *= 0.
            
            l_.append(layer)
            r0_.append(res0)
            r1_.append(res1)

        self.layers.clear()
        self.residuals_0.clear()
        self.residuals_1.clear()
        self.residuals_dim.clear()
        self.layers_dim.clear()

        self.layers = l_
        self.residuals_0 = r0_
        self.residuals_1 = r1_
        self.residuals_dim = new_residuals_dim
        self.layers_dim = new_layers_dim
        return

    ############# Network Increase Operation
    ############# ShortCut connection Removal

    
