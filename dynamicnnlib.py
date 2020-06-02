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

        self.trainable_layers = []
        self.removable_layers = []

        self.decay_steps = 0
        self.neuron_decay_steps = 0

        self.weight_decay_rates = None

        self.decay = {}
        self.decay_res = {}    
        self.significance = None
        self.significance_res = None

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

            self.trainable_layers.append(i)

    ### Print and Repr Methods ###
    def __repr__(self):
        str_up = ""
        for i in range(len(self.layers_dim)):
            str_up += f"{' '*(2-len(str(self.layers_dim[i])))}{self.layers_dim[i]} "
            if i < len(self.layers_dim)-1:
                str_up += "=="
            else:
                str_up += "->"
        str_mid = " "
        for i in range(len(self.residuals_dim)):
            str_mid += f" \\  /"
        
        str_bot = " "
        for i in range(len(self.residuals_dim)):
            str_bot += f"  {' '*(2-len(str(self.residuals_dim[i])))}{self.residuals_dim[i]} "

        toret = str_up+"\n"+str_mid+"\n"+str_bot
        return toret
    ### Print and Repr Methods ###

    ############# Basic Network Operations
    
    def forward(self, input):
        self.decay_removable_weights()
        self._decay_removable_neurons_()
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

        self._freeze_removable_neurons_()
        return del_input

    def update(self):
        for i in self.trainable_layers:
            self.layers[i].update()

        for i in range(len(self.layers)):
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


        old_num_neurons = sum(self.layers_dim) + sum(self.residuals_dim)
        new_num_neurons = sum(new_layers_dim) + sum(new_residuals_dim)
        neurons_added = new_num_neurons - old_num_neurons

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
        
        return neurons_added
    ############# Network Increase Operation
    ############# ShortCut connection Removal

    def _identify_removable_shortcut_connections_(self):
        self.removable_layers = []
        self.trainable_layers = []
        for i in range(len(self.layers)):
            h = self.residuals_0[i].weights.shape[1]
            io = self.layers[i].weights.shape
            if h > np.min(io): #+1:
                self.removable_layers.append(i)
            else:
                self.trainable_layers.append(i)
        pass

    def _set_decay_rate_(self, decay_steps):
        self.decay_steps = decay_steps
        self.weight_decay_rates = []
        for rl in self.removable_layers:
            w_dr = self.layers[rl].weights/decay_steps
            self.weight_decay_rates.append(w_dr)
        pass    
        
    def remove_decayed_layer(self):
        ### Removable layers are decayed to zero in this case
        self.removable_layers.sort()    
        for i, rl in enumerate(self.removable_layers):
            ### this is to correct the removable layer position that changes due to previous removal of decayed layer 
            ## previously, one is removed and two are added (-1+2=1).
            rl = rl+i
            ### add the bias of the removable layer to the residual out layer
            self.residuals_1[rl].bias += self.layers[rl].bias
            self.layers[rl].bias *= 0.
            
            ### Change the network structure
            ## Make the residuals layer, normal layers
            ## add residual layer to the newly added layers

            del self.layers[rl]
            l0 = tnn.LinearLayer(0, 0,
                                weights=self.residuals_0[rl].weights,
                                bias=self.residuals_0[rl].bias,
                                optimizer=self.optimizer)
            relu = Relu_1Linear()
            l1 = tnn.LinearLayer(0, 0,
                                weights=self.residuals_1[rl].weights,
                                bias=self.residuals_1[rl].bias,
                                optimizer=self.optimizer)
            res0_l0 = tnn.NonLinearLayer(l0.weights.shape[0],
                                    1,
                                    activation=Relu_1Linear(),
                                    optimizer=self.optimizer)
            res1_l0 = tnn.LinearLayer(1,
                                l0.weights.shape[1],
                                optimizer=self.optimizer)
            res1_l0.weights *= 0.
            res0_l1 = tnn.NonLinearLayer(l1.weights.shape[0],
                                    1,
                                    activation=Relu_1Linear(),
                                    optimizer=self.optimizer)
            res1_l1 = tnn.LinearLayer(1,
                                l1.weights.shape[1],
                                optimizer=self.optimizer)
            res1_l1.weights *= 0.
            
            
            
            self.layers.insert(rl, l0)
            self.layers.insert(rl+1, l1)
            self.relus.insert(rl, relu)
            self.layers_dim.insert(rl+1, l0.weights.shape[1])
            
            del self.residuals_0[rl]
            del self.residuals_1[rl]
            self.residuals_0.insert(rl ,res0_l0)
            self.residuals_0.insert(rl+1 ,res0_l1)
            self.residuals_1.insert(rl ,res1_l0)
            self.residuals_1.insert(rl+1 ,res1_l1)
            
            del self.residuals_dim[rl]
            self.residuals_dim.insert(rl, 1)
            self.residuals_dim.insert(rl+1, 1)
            
            self.removable_layers = []
        return

    def decay_removable_weights(self):
        if len(self.removable_layers) == 0:
            return

        if self.decay_steps > 0:
            for i, rl in enumerate(self.removable_layers):
                self.layers[rl].weights -= self.weight_decay_rates[i]
            self.decay_steps -= 1
        else:
            self.weight_decay_rates = None
            self.remove_decayed_layer()
        pass

    def start_decaying_removable_shortcut_connections(self, decay_steps=500):
        self._identify_removable_shortcut_connections_()
        self._set_decay_rate_(decay_steps)


    ############# ShortCut connection Removal
    ############# Neuron Importance and Removal
    def compute_neuron_significance(self, dataX, batch_size=None):
        assert len(self.layers) == len(self.residuals_0)
        
        # sum_activation = [0]*(len(self.layers))
        count_non_zero = [0]*(len(self.layers))
        std_zee = [0]*(len(self.layers))
        
        # sum_activation_res = [0]*(len(self.residuals_0))
        count_non_zero_res = [0]*(len(self.residuals_0))
        std_zee_res = [0]*(len(self.residuals_0))
        
        
        data_size = len(dataX)

        ## do computation on batch wise manner
        if batch_size is None:
            batch_size = data_size
        start = np.arange(0, data_size, batch_size)
        stop = start+batch_size
        if stop[-1]>data_size:
            stop[-1] = data_size

        ### Compute average over the batch
        for idx in range(len(start)):
            activations = dataX[start[idx]:stop[idx]]
            for i in range(len(self.layers)):
                out0 = self.layers[i].forward(activations)
                h0 = self.residuals_0[i].forward(activations)
                h1 = self.residuals_1[i].forward(h0)
                activations = self.relus[i].forward(out0+h1)

                # sum_actv = activations.sum(axis=0, keepdims=True)
                if i == len(self.layers)-1:
                    std_z = activations.std(axis=0, keepdims=True)
                else:
                    std_z = self.relus[i].x.std(axis=0, keepdims=True)
                count_actv = (activations > 0).astype(float).sum(axis=0, keepdims=True)
                # sum_activation[i] += sum_actv
                count_non_zero[i] += count_actv
                std_zee[i] += std_z
                
                # sum_actv_res = h0.sum(axis=0, keepdims=True)
                std_z_res = self.residuals_0[i].zee.std(axis=0, keepdims=True)
                count_actv_res = (h0 > 0).astype(float).sum(axis=0, keepdims=True)
                # sum_activation_res[i] += sum_actv_res
                count_non_zero_res[i] += count_actv_res
                std_zee_res[i] += std_z_res
                
        for i in range(len(count_non_zero)):
            # sum_activation[i] /= data_size
            count_non_zero[i] /= data_size
            std_zee[i] /= len(start)
            # sum_activation_res[i] /= data_size
            count_non_zero_res[i] /= data_size
            std_zee_res[i] /= len(start)
        
        # mean_activation = sum_activation
        prob_non_zero = count_non_zero

        # mean_activation_res = sum_activation_res
        prob_non_zero_res = count_non_zero_res

        ### Compute the significance based on NISP and probability of activation
        significance = []
        significance_res = []
            
        sig_prev = np.ones([1,self.layers[-1].weights.shape[1]])
        for i in reversed(range(len(self.layers))):
        
            fac_res1 = np.abs(self.residuals_1[i].weights.T)
            sig_res1_ = sig_prev @ fac_res1 ### significance before residuals_1[i]
    #         sig_res1 = sig_res1 * mean_activation_res[i].reshape(1,-1)
            sig_res1 = sig_res1_ * prob_non_zero_res[i]
            
            
            fac_res0 = np.abs(self.residuals_0[i].weights.T)
            sig_res0 = sig_res1 @ fac_res0 ### significance before residuals_0[i]
            
            fac = np.abs(self.layers[i].weights).T
            sig_lin = sig_prev @ fac ### significance before the linear
            
            sig_ = sig_lin + sig_res0 ### significance is gathered from linear and residual
    #         sig = sig * mean_activation[i-1].reshape(1,-1)
            sig = sig_ * prob_non_zero[i-1]


            significance.append(sig_)
            significance_res.append(sig_res1_)
            sig_prev = sig
            
        significance.reverse()
        significance_res.reverse()
        
        ### rescaling significance by neuron's standard deviation
        ### and scale for division of space.. (half will get maximum value = 1)
        
        for i in range(len(self.layers)):
            # scale_res = np.where(prob_non_zero_res[i]<0.5, 1, 1/(prob_non_zero_res[i]+0.001)*(1-prob_non_zero_res[i]))
            scale_res = 2*np.minimum(prob_non_zero_res[i], 1 - prob_non_zero_res[i])            
            scale_res *= std_zee_res[i]
            significance_res[i] *= scale_res
            ### the significance of the linear part should be high (cannot be deleted to bbe on safe side)
            significance_res[i][0,0] = 1.
            
            if i == len(self.layers)-1: break
            # scale = np.where(prob_non_zero[i]<0.5, 1, 1/prob_non_zero[i]*(1-prob_non_zero[i]))
            scale = 2*np.minimum(prob_non_zero[i], 1 - prob_non_zero[i])
            scale *= std_zee[i]
            significance[i+1] *= scale 
            significance[i+1][0,0] = 1.
            ### the significance of neurons preserving the dimensions must be high
            
        del significance[0]
        del prob_non_zero[-1]

        self.significance = significance
        self.significance_res = significance_res
        # return significance, significance_res

    def _sort_neuron_significance_(self):

        all_sig = np.concatenate(self.significance, axis=1).reshape(-1)
        all_sig_res = np.concatenate(self.significance_res, axis=1).reshape(-1)
        
        residual_split_indx = len(all_sig)
        alls = np.concatenate([all_sig, all_sig_res])
        si_ = np.argsort(alls)
        si = np.argsort(si_)
        
        si, si_res = np.split(si, [residual_split_indx])
        
        ## last layer and first is not in sig, last of remaining is not needed for split 
        sort_sig = np.split(si, np.cumsum(self.layers_dim[1:-2]))
        ## last remaining dim is not needed for split 
        sort_sig_res = np.split(si_res, np.cumsum(self.residuals_dim[:-1]))

        return sort_sig, sort_sig_res
    

    def _identify_decayable_neurons_(self, decay_n, threshold = 0.02):
        importance, importance_res = self._sort_neuron_significance_()
        dec_neurons_res = {}
        dec_neurons = {}
        sig = self.significance
        sig_res = self.significance_res
        for i in range(len(self.layers)):
            ## for residual layers
            unimp = (importance_res[i] < decay_n).astype(int)
            ## neuron is important if its  importance is more than threshold
            unimp[sig_res[i].reshape(-1) > threshold] *= 0
            unimp = np.nonzero(unimp)[0]
            if len(unimp)>0:
                dec_neurons_res[i] = unimp
            
            if i == len(self.layers)-1: break
            ## for normal layers
            unimp = (importance[i] < decay_n).astype(int)
            unimp[sig[i].reshape(-1) > threshold] *= 0
            unimp_ = np.nonzero(unimp)[0]
            ## the layers must have enough dimension to preserve the dimension of incoming&outgoing
            min_dim = min(self.layers_dim[i], self.layers_dim[i+2])
            if self.layers_dim[i+1]-len(unimp_) < min_dim:
                keep_indx = np.argsort(sig[i].reshape(-1))[-min_dim:]
                unimp[keep_indx] *= 0
                unimp_ = np.nonzero(unimp)[0]
                
            if len(unimp_)>0:
                dec_neurons[i] = unimp_
        
        self.decay = dec_neurons
        self.decay_res = dec_neurons_res
        # return dec_neurons, dec_neurons_res

    
    def _set_neuron_decay_rate_(self, steps=500):
        self.neuron_decay_rate = {}
        self.neuron_decay_rate2 = {}
        self.neuron_res_decay_rate = {}
        self.neuron_decay_steps = steps
        for li, neurons in self.decay.items():
            self.neuron_decay_rate[li] = self.layers[li+1].weights[neurons]/steps
            self.neuron_decay_rate2[li] = self.residuals_0[li+1].weights[neurons]/steps
        for rli, neurons in self.decay_res.items():
            self.neuron_res_decay_rate[rli] = self.residuals_1[rli].weights[neurons]/steps
        pass

    def _decay_removable_neurons_(self):
        if self.neuron_decay_steps > 0:
            for li, neurons in self.decay.items():
                self.layers[li+1].weights[neurons] -= self.neuron_decay_rate[li]
                self.residuals_0[li+1].weights[neurons] -= self.neuron_decay_rate2[li]
            for rli, neurons in self.decay_res.items():
                self.residuals_1[rli].weights[neurons] -= self.neuron_res_decay_rate[rli]
            self.neuron_decay_steps -= 1
            
            if self.neuron_decay_steps == 0:
                self._remove_decayed_neurons_()
                self.significance = None
                self.significance_res = None
                self.decay = {}
                self.decay_res = {}    
            
        pass

    def _freeze_removable_neurons_(self):
        for li, neurons in self.decay.items():
            self.layers[li+1].del_weights[neurons] *= 0.
            self.residuals_0[li+1].del_weights[neurons] *= 0.
            self.layers[li].del_weights[:, neurons] *= 0.
                
        for rli, neurons in self.decay_res.items():
            self.residuals_1[rli].del_weights[neurons] *= 0.
            self.residuals_0[rli].del_weights[:, neurons] *= 0.
                
        pass

    def _remove_decayed_neurons_(self):
        for li, neurons in self.decay.items():
            self.layers[li+1].weights = np.delete(self.layers[li+1].weights, (neurons), axis=0)
            self.layers[li+1].weightsOpt = self.optimizer.set_parameter(self.layers[li+1].weights)
            
            self.residuals_0[li+1].weights = np.delete(self.residuals_0[li+1].weights, (neurons), axis=0)
            self.residuals_0[li+1].weightsOpt = self.optimizer.set_parameter(self.residuals_0[li+1].weights)

            self.layers[li].weights = np.delete(self.layers[li].weights, (neurons), axis=1)
            self.layers[li].weightsOpt = self.optimizer.set_parameter(self.layers[li].weights)

            self.layers[li].bias = np.delete(self.layers[li].bias, (neurons), axis=0)
            self.layers[li].biasOpt = self.optimizer.set_parameter(self.layers[li].bias)
            
            self.residuals_1[li].weights = np.delete(self.residuals_1[li].weights, (neurons), axis=1)
            self.residuals_1[li].weightsOpt = self.optimizer.set_parameter(self.residuals_1[li].weights)

            self.residuals_1[li].bias = np.delete(self.residuals_1[li].bias, (neurons), axis=0)
            self.residuals_1[li].biasOpt = self.optimizer.set_parameter(self.residuals_1[li].bias)

            self.layers_dim[li+1] -= len(neurons)
            
        for rli, neurons in self.decay_res.items():
            self.residuals_1[rli].weights = np.delete(self.residuals_1[rli].weights, (neurons), axis=0)
            self.residuals_1[rli].weightsOpt = self.optimizer.set_parameter(self.residuals_1[rli].weights)

            self.residuals_0[rli].weights = np.delete(self.residuals_0[rli].weights, (neurons), axis=1)
            self.residuals_0[rli].weightsOpt = self.optimizer.set_parameter(self.residuals_0[rli].weights)
            
            self.residuals_0[rli].bias = np.delete(self.residuals_0[rli].bias, (neurons), axis=0)
            self.residuals_0[rli].biasOpt = self.optimizer.set_parameter(self.residuals_0[rli].bias)

            self.residuals_dim[rli] -= len(neurons)
        pass

    def start_decaying_less_significant_neurons(self, decay_n, threshold=0.02, steps=1000):
        assert self.significance_res is not None

        self._identify_decayable_neurons_(decay_n, threshold)
        self._set_neuron_decay_rate_(steps)

    ############# Neuron Importance and Removal





    
