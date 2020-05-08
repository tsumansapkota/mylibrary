import numpy as np

try:
    import nnlib as tnn
except ImportError:
    import mylibrary.nnlib as tnn
    pass

class DynamicDoubleReluNN(object):

    def __init__(self, input_dim, output_dim, optimizer = tnn.Adam(), max_width_factor = 2, ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.optimizer = optimizer
        self.max_width_factor = max_width_factor

        self.layers= [tnn.LinearLayer(self.input_dim, self.output_dim, optimizer=self.optimizer)]

        self.autoform = tnn.AutoForm(new_layers=True)
        self.autoform.layerList = self.layers
        

    ### BASIC NEURAL NETWORK FUNCTIONS ###
    def forward(self, inputs):
        outputs  = self.autoform.forward(inputs)
        return outputs

    def backward(self, del_outputs):
        del_inputs = self.autoform.backward(del_outputs)
        return del_inputs

    def update(self):
        self.autoform.update()
    ### BASIC NEURAL NETWORK FUNCTIONS ###

    ### DYNAMIC NEURAL NETWORK FUNCTIONS ###
    def 
    