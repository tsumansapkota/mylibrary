import math
import numpy
import nnlib as tnn

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

    def __init__(self, position:tuple, activation = tnn.Relu(), max_synapse = 20 ):
        self.position = position
        self.output = None
        self.activation = activation
        self.max_synapse = max_synapse
        
        ### Synapse
        self.weights = None
        self.synapse = []



class NeuralNetwork(object):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.

    