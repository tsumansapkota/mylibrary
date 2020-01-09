import math
import numpy as np
import mylibrary.nnlib as tnn

'''
Neurons:
Sensor >> Input_Neuron
    -- Only Outgoing Connections
    -- Does not have Associated weights
Motor >> Output_Neuron
    -- Only Incoming Connections
    -- Incoming Weights Associated
Inter >> Processor_Neuron
    -- Both Incoming and Outgoing Connections
'''

class Neuron(object):

    def __init__(self, position:tuple, activation = tnn.Linear(), max_synapse = 20 ):
        self.position = position
        self.bias = 0.
        self.del_bias = 0.
        self.activation = activation
        self.max_synapse = max_synapse
        
        ### Synapse
        self.weights = None
        self.synapses = []
        self.del_weights = None

        ### Propagation
        self.outputs = None#[0]
        self.del_outputs = None


    def _initialize_connection_(self, synapses:list):
        self.synapses = synapses
        self.weights = np.random.normal(size=len(synapses))

    def forward(self):
        self.outputs = 0.
        for i in range(len(self.synapses)):
            self.outputs += (self.synapses[i].outputs * self.weights[i])
        self.outputs += self.bias
        self.outputs = self.activation.forward(self.outputs)

    def backward(self):
        del_zee = self.activation.backward(self.del_outputs)
        self.del_bias = np.mean(del_zee)

        self.del_weights = np.zeros_like(self.weights)
        for i in range(len(self.synapses)):
            self.del_weights[i] =  np.mean(del_zee*self.synapses[i].outputs)
            self.synapses[i].del_outputs = del_zee*self.weights[i]

    def update(self, learning_rate):
        self.bias -= learning_rate*self.del_bias
        self.weights -= learning_rate*self.del_weights
            


        

class NeuralNetwork(object):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.sensors = [Neuron(position=(0.0,0.5), activation=tnn.Linear()) for _ in range(input_dim)]
        self.motors = [Neuron(position=(1.0,0.5), activation=tnn.Linear()) for _ in range(output_dim)]
        self.inners = [] ## List of neurons

        self.inputs = None
        self.outputs = None
        self._initialize_connection_()

    def _initialize_connection_(self):
        for i in range(self.output_dim):
            self.motors[i]._initialize_connection_(self.sensors)
        # if len(self.inners) > 0 :
        #     for i in range(len(self.inners)):
        #     self.inners[i]._initialize_connection_(self.sensors)
            
        

    def _sort_neurons_by_xposition_(self):
        '''
        https://stackoverflow.com/questions/403421/how-to-sort-a-list-of-objects-based-on-an-attribute-of-the-objects
        '''
        if len(self.inners) > 0 :
            self.inners.sort(key=lambda x: x.position[0])
        return

    def forward(self, inputs):
        assert inputs.shape[1] == self.input_dim
        self.inputs = inputs
        ## Setting sensors output to given input
        for i in range(self.input_dim):
            self.sensors[i].outputs = self.inputs[:,i]
        ## Sorting the neurons,, preparation for forward propagation
        self._sort_neurons_by_xposition_()
        
        ## exact forward propagation
        for neuron in self.inners+self.motors:
            neuron.forward()

        self.outputs = np.array([self.motors[i].outputs for i in range(self.output_dim)]).T
        return self.outputs


    def backward(self, del_outputs):
        self.del_outputs = del_outputs
        for i in range(len(self.motors)):
            self.motors[i].del_outputs = self.del_outputs[:,i]

        for neuron in reversed(self.inners+self.motors):
            neuron.backward()

        del_inputs = np.array([self.sensors[i].del_outputs for i in range(self.input_dim)]).T
        return del_inputs

    def update(self, learning_rate=0.01):
        for neuron in self.inners+self.motors:
            neuron.update(learning_rate)
