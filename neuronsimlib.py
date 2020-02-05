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

    def __init__(self, position:tuple, activation = tnn.Linear(), optimizer=tnn.SGD(), max_synapse = 20 ):
        self.position = np.array(position)
        self.cpg = 1 ## count position
        self.del_position = np.zeros_like(self.position)
        self.bias = 0.
        self.del_bias = 0.
        self.activation = activation
        self.max_synapse = max_synapse
        
        ### Synapse
        self.weights = 0.
        self.synapses = []
        self.diff = []
        self.idist = []
        self.del_weights = 0.

        ### Propagation
        self.outputs = None#[0]
        self.del_outputs = 0


    def _initialize_connection_(self, synapses:list):
        self.synapses = synapses
        m = len(synapses)
        self.weights = np.random.normal(size=m)
        self.diff = [0]*m
        self.idist = [0]*m

    def forward(self):
        self.outputs = 0.
        for i in range(len(self.synapses)):
            self.diff[i] = self.position - self.synapses[i].position
            self.idist[i] = 1/(np.sqrt((self.diff[i]**2).sum()))

            self.outputs += (self.synapses[i].outputs * self.weights[i] * self.idist[i])

        self.outputs += self.bias
        self.outputs = self.activation.forward(self.outputs)

    def backward(self):
        del_zee = self.activation.backward(self.del_outputs)
        self.del_bias = np.mean(del_zee)

        self.del_weights = np.zeros_like(self.weights)
        for i in range(len(self.synapses)):
            self.del_weights[i] =  np.mean(del_zee*self.synapses[i].outputs) * self.idist[i]
            self.synapses[i].del_outputs += del_zee * self.weights[i] * self.idist[i]
            
            del_position_i = self.diff[i]/np.power(self.idist[i], 3.) * np.mean(del_zee*self.synapses[i].outputs)*self.weights[i]
            self.del_position += del_position_i
            self.synapses[i].del_position -= del_position_i
            self.cpg += 1
            self.synapses[i].cpg += 1


    def update_params(self, learning_rate):
        self.bias -= learning_rate*self.del_bias
        self.weights -= learning_rate*self.del_weights
        self.del_bias *= 0.
        self.del_weights *= 0.
        self.del_outputs *= 0.
            
    def update_position(self, learning_rate):
        self.position -= learning_rate*self.del_position #/ self.cpg
        self.del_position *= 0.
        self.cpg = 1

        

class NeuralNetwork(object):

    def __init__(self, input_dim, output_dim, hidden_neurons=4, activation_class=tnn.Relu, optimizer=tnn.Adam()):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_neurons = hidden_neurons
        self.activation = activation_class

        self.sensors = [Neuron(position=(0.0,0.5), activation=tnn.Linear(), optimizer=optimizer) for _ in range(input_dim)]
        self.motors = [Neuron(position=(1.0,0.5), activation=tnn.Linear(), optimizer=optimizer) for _ in range(output_dim)]
        self.inners = [] ## List of neurons

        self.inputs = None
        self.outputs = None
        self._initialize_connection_1()

        self.update_count = 0

    def _initialize_connection_(self):
        for i in range(self.output_dim):
            self.motors[i]._initialize_connection_(self.sensors)
        # if len(self.inners) > 0 :
        #     for i in range(len(self.inners)):
        #     self.inners[i]._initialize_connection_(self.sensors)
            
    def _initialize_connection_1(self):
        for i in range(self.hidden_neurons):
            # randy = np.random.uniform()
            randy = np.random.uniform(0.4,0.6)
            # randy = 0.
            inner = Neuron(position=(0.7,randy), activation=self.activation())
            inner._initialize_connection_(self.sensors)
            self.inners.append(inner)
        for i in range(self.output_dim):
            self.motors[i]._initialize_connection_(self.inners)
        

        

        

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

    def update(self, learning_rate=0.03):
        # if self.update_count %2 == 0:
        #     for neuron in self.inners+self.motors:
        #         neuron.update_params(learning_rate)
        # else:
        #     for neuron in self.sensors+self.inners+self.motors:
        #         neuron.update_position(learning_rate)
        # self.update_count += 1

        for neuron in self.inners+self.motors:
            neuron.update_params(learning_rate)
        for neuron in self.sensors+self.inners+self.motors:
            neuron.update_position(learning_rate)
        
        # for neuron in self.motors+self.inners:
        #     neuron.update_params(learning_rate)

        self.reposition()

    def reposition(self):
        # (0.0,0.5)(1.0,0.5)
        
        pass
