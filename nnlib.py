import abc
import numpy as np

layerList = []


##############################################################

class LossFunction(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def loss(output, target):
        pass

    @staticmethod
    @abc.abstractmethod
    def del_loss(output, target):
        pass


class MseLoss(LossFunction):
    @staticmethod
    def loss(output, target):
        # return 0.5 * np.mean((output - target) ** 2, axis=None)  ## should the axis be None instead of 1??
        return 0.5 * ((output - target) ** 2).mean()  ## should the axis be None instead of 1??

    @staticmethod
    def del_loss(output, target):
        return output - target


class CrossEntropyBinary(LossFunction):
    @staticmethod
    def loss(output, target):
        # return -np.mean(np.sum(target * np.log(output) + (1 - target) * np.log(1 - output), axis=1))
        return -((target * np.log(output) + (1 - target) * np.log(1 - output)).sum(axis=1)).mean()

    @staticmethod
    def del_loss(output, target):
        return output - target


class SoftmaxCrossEntropy(LossFunction):
    @staticmethod
    def loss(output, target, epsilon=1e-11):
        # exps = np.exp(output - np.max(output)) ## for stability
        # Prevent overflow
        # output = np.clip(output, epsilon, 1 - epsilon)
        output = output.clip(epsilon, 1 - epsilon)  # idea from internet
        # exps = np.exp(output)
        exps = np.e ** output
        # probs = exps / np.sum(exps)  #####
        probs = exps / exps.sum()
        # return -((target * np.log(probs)).sum(axis=1)).mean()
        return -((target * np.log(probs)).sum(axis=1)).mean()

    @staticmethod
    def del_loss(output, target, epsilon=1e-11):
        # output = np.clip(output, epsilon, 1 - epsilon)
        output = output.clip(epsilon, 1 - epsilon)
        # divisor = np.maximum(output * (1 - output), epsilon)
        divisor = 1.
        return (output - target) / divisor


class CrossEntropyLoss(LossFunction):
    # softmax layer must be implemented before the this loss function
    @staticmethod
    def loss(output, target, epsilon=1e-11):
        output = output+epsilon
        return -((target * np.log(output)).sum(axis=1)).mean()

    @staticmethod
    def del_loss(output, target, epsilon=1e-11):
        return output - target


##############################################################
##############################################################

class Logits():
    @staticmethod
    def index_to_logit(index, length=None):
        index = index.astype(int)
        size = len(index)
        if length is None:
            length = np.max(index) + 1
        logits = np.zeros([size, length])
        # logits[range(size), logits.astype(int)] = 1
        for indx in range(size):
            logits[indx, index[indx]] = 1
        return logits

    @staticmethod
    def logit_to_index(logits):
        return np.argmax(logits, axis=1)


##############################################################
##############################################################


class Layer(abc.ABC):

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def backward(self, output_delta):
        pass

    def update(self):
        pass


class Linear(Layer):

    def __init__(self):
        # print("linear index = ", len(layerList))
        layerList.append(self)


    def forward(self, x):
        return x

    def backward(self, output_delta):
        return output_delta


class Sigmoid(Layer):
    def __init__(self):
        layerList.append(self)
        self.out = None
        self.del_output = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, output_delta):
        self.del_output = output_delta
        return (self.out * (1 - self.out)) * output_delta


class Tanh(Layer):
    def __init__(self):
        layerList.append(self)
        self.out = None
        self.del_output = None


    def forward(self, x):
        self.out = 2 / (1 + np.exp(-2 * x)) - 1
        return self.out

    def backward(self, output_delta):
        self.del_output = output_delta
        return (1 - self.out ** 2) * output_delta


class Relu(Layer):
    def __init__(self):
        layerList.append(self)
        self.x = None
        self.del_output = None


    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, output_delta):
        self.del_output = output_delta
        # print((self.x >= 0).astype(float))
        return ((self.x >= 0).astype(float)) * output_delta


class LeakyRelu(Layer):
    def __init__(self, alpha=0.01, ):
        layerList.append(self)
        self.x = None
        self.alpha = alpha
        self.del_output = None

    def forward(self, x):
        self.x = x
        toret = np.maximum(self.alpha * x, x)
        # print('toret=',toret)
        return toret

    def backward(self, output_delta):
        self.del_output = output_delta
        dx = np.ones_like(self.x)
        dx[self.x < 0] = self.alpha
        return dx * output_delta


class Softmax(Layer):
    def __init__(self, ):
        layerList.append(self)
        self.out = None
        self.del_output = None


    def forward(self, x):
        # for stability
        # x = x - x.max()  # the output of softmax doesnot change by shifting the inputs
        # exp = np.exp(x)
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.out = exp / np.sum(exp, axis=1, keepdims=True)
        return self.out

    def backward(self, output_delta):
        self.del_output = output_delta
        return (self.out * (1 - self.out)) * output_delta
    

class Negative(Layer):
    def __init__(self):
        layerList.append(self)
        self.x = None
        self.del_output = None

    def forward(self, x):
        self.x = x
        return -x

    def backward(self, output_delta):
        self.del_output = output_delta
        return -output_delta

    
class DoubleRelu(Layer):
    def __init__(self):
        layerList.append(self)
        self.x = None
        self.output = None
        self.del_output = None
    
    def forward(self, x):
        self.x = x
        xp = np.maximum(x, 0)
        xn = np.minimum(x, 0)
        self.output = np.concatenate((xp, xn), axis=1)
        return self.output

    def backward(self, output_delta):
        self.del_output = output_delta
        # print(output_delta.shape)
        # print(self.output.shape)
        # print(self.x.shape)
        deltap, deltan = np.split(output_delta, 2, axis=1)
        # print(deltap.shape, deltan.shape)
        toret = np.where(self.x>=0, deltap, deltan)
        # print(toret.shape)
        return toret


        

##############################################################
##############################################################

class Optimizer(abc.ABC):

    # @abc.abstractmethod
    # def __init__(self):
    #     pass

    @abc.abstractmethod
    def set_parameter(self, parameter):
        pass

    @abc.abstractmethod
    def compute_gradient(self, var, grad):
        pass


class SGD(Optimizer):
    class Parameter:
        def __init__(self, optimizer, parameters):
            self.my_optimizer = optimizer

        def compute_gradient(self, grad, new_learning_rate=None):
            if new_learning_rate is not None: self.my_optimizer.learning_rate = new_learning_rate
            return self.my_optimizer.compute_gradient(self, grad)

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.parameter_list = []
        pass

    def set_parameter(self, parameters):
        parameter = self.Parameter(self, parameters)
        self.parameter_list.append(parameter)
        return parameter

    def compute_gradient(self, var, grad):
        return grad * self.learning_rate


class Momentum(Optimizer):
    class Parameter:
        def __init__(self, optimizer, parameters):
            self.my_optimizer = optimizer
            self.velocity = np.zeros_like(parameters)
            pass

        def compute_gradient(self, grad, new_learning_rate=None):
            if new_learning_rate is not None: self.my_optimizer.learning_rate = new_learning_rate
            return self.my_optimizer.compute_gradient(self, grad)

    def __init__(self, learning_rate=0.01, beta1=0.9):
        self.learning_rate = learning_rate
        self.beta = beta1
        self.parameter_list = []

    def set_parameter(self, parameters):
        """
        :param parameters:
        :return: return parameter pointer
        """
        parameter = self.Parameter(self, parameters)
        self.parameter_list.append(parameter)
        return parameter

    def compute_gradient(self, var, grad):
        var.velocity = self.beta * var.velocity + (1 - self.beta) * grad
        return var.velocity * self.learning_rate


class RMSProp(Optimizer):
    class Parameter:
        def __init__(self, optimizer, parameters):
            self.my_optimizer = optimizer
            self.sq_grad = np.zeros_like(parameters)
            pass

        def compute_gradient(self, grad, new_learning_rate=None):
            if new_learning_rate is not None: self.my_optimizer.learning_rate = new_learning_rate
            return self.my_optimizer.compute_gradient(self, grad)

    def __init__(self, learning_rate=0.01, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta = beta2
        self.epsilon = epsilon
        self.parameter_list = []

    def set_parameter(self, parameters):
        """
        :param parameters:
        :return: return parameter pointer
        """
        parameter = self.Parameter(self, parameters)
        self.parameter_list.append(parameter)
        return parameter

    def compute_gradient(self, var, grad):
        var.sq_grad = self.beta * var.sq_grad + (1 - self.beta) * np.square(grad)
        return grad / np.sqrt(var.sq_grad + self.epsilon) * self.learning_rate


class Adam(Optimizer):  # rmsprop + momentum
    class Parameter:
        def __init__(self, optimizer, parameters):
            self.my_optimizer = optimizer
            self.grad_velocity = np.zeros_like(parameters)
            self.squared_grad_vel = np.zeros_like(parameters)
            self.count = 1
            pass

        def compute_gradient(self, grad, new_learning_rate=None):
            if new_learning_rate is not None: self.my_optimizer.learning_rate = new_learning_rate
            return self.my_optimizer.compute_gradient(self, grad)

    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.parameter_list = []

    def set_parameter(self, parameters):
        """
        :param parameters:
        :return: return parameter pointer
        """
        parameter = self.Parameter(self, parameters)
        self.parameter_list.append(parameter)
        return parameter

    def compute_gradient(self, var, grad):
        # exponentially weighted average of gradients
        var.grad_velocity = self.beta1 * var.grad_velocity + \
                            (1 - self.beta1) * grad
        grad_corrected = var.grad_velocity / (1 - self.beta1 ** var.count)  # bias correction

        # exponentially weighted average of squared gradients
        var.squared_grad_vel = self.beta2 * var.squared_grad_vel + \
                               (1 - self.beta2) * np.square(grad)
        sq_grad_corrected = var.squared_grad_vel / (1 - np.power(self.beta2, var.count))  # bias correction

        var.count += 1
        overall_grad = grad_corrected / np.sqrt(
            sq_grad_corrected + self.epsilon)  # combined direction with gradient corrected
        # overall_grad = self.grad_velocity / np.sqrt(self.squared_grad_vel + self.epsilon)  # combined direction
        return overall_grad * self.learning_rate


##############################################################
##############################################################

class DoubleReluLayer(Layer):
    def __init__(self, input_dim, weights=None, bias=None, optimizer=SGD()):
        if weights is None:
            # self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)  # Xavier/He initialization
            self.weights = np.random.randn(2, input_dim)+1  # normal distribution of linear slope
        else:
            self.weights = weights

        if bias is None:
            self.bias = np.zeros(input_dim)
        else:
            self.bias = bias

        # print(self.weights.shape, self.weights)

        self.weightsOpt = optimizer.set_parameter(self.weights)
        self.biasOpt = optimizer.set_parameter(self.bias)

        layerList.append(self)

        self.input = None
        self.mask = None
        self.output = None

        self.del_weights = None
        self.del_bias = None
        self.del_output = None

    def forward(self, input):
        self.input = input
        self.mask = (input >= 0)
        # print(self.index)
        # print(self.index.shape)
        self.output = input * np.where(self.mask, self.weights[1], self.weights[0]) + self.bias
        return self.output

    def backward(self, output_delta):
        self.del_output = output_delta
        self.del_bias = np.mean(self.del_output, axis=0)

        dw0 = np.where(self.mask, 0, self.input)*output_delta
        dw1 = np.where(self.mask, self.input, 0)*output_delta
        self.del_weights = np.array([dw0.mean(axis=0), dw1.mean(axis=0)])

        return np.where(self.mask, self.weights[1], self.weights[0]) * output_delta

    def update(self):
        gradients = self.weightsOpt.compute_gradient(self.del_weights)
        self.weights -= gradients

        gradients = self.biasOpt.compute_gradient(self.del_bias)
        self.bias -= gradients


class LinearLayer(Layer):
    """
        Linear Layer class
        Z = X . W           ; X -> R(num_samples , input_dim)
                            ; W -> R(input_dim , output_dim)
        Y = actv(Z)

        given: dE/dY  (AKA dY)      ; R(num_samples, output_dim)
        dE/dZ = dY * actv'(Z)       ; ''''''''''''''''''''''''''
        dE/dW = dE/dZ . dZ/dW
        .: dW = X.T . dZ    ; R(input_dim, output_dim)
        .: dX = dZ . W.T      ; R(num_samples, input_dim)
        """

    def __init__(self, input_dim, output_dim, weights=None, bias=None, optimizer=SGD()):
        if weights is None:
            # self.weights = np.random.randn(input_dim, output_dim) * 2 / (input_dim + output_dim)
            # self.weights = (np.random.randn(input_dim, output_dim) * 0.2) - 0.1
            self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)  # Xavier/He initialization
        else:
            self.weights = weights

        if bias is None:
            self.bias = np.zeros(output_dim)
        else:
            self.bias = bias

        self.weightsOpt = optimizer.set_parameter(self.weights)
        self.biasOpt = optimizer.set_parameter(self.bias)

        self.input = None
        self.zee = None

        self.del_weights = None
        self.del_bias = None
        self.del_zee = None

        layerList.append(self)
        pass

    def forward(self, input):
        self.input = input
        self.zee = input @ self.weights + self.bias  # @ == .dot
        return self.zee

    def backward(self, output_delta):
        self.del_zee = output_delta
        m = output_delta.shape[0]
        self.del_bias = np.mean(self.del_zee, axis=0)
        self.del_weights = self.input.T.dot(self.del_zee) * 1 / m
        return self.del_zee.dot(self.weights.T)

    def update(self):
        gradients = self.weightsOpt.compute_gradient(self.del_weights)
        self.weights -= gradients  # * learning_rate
        # print(self.bias.shape, self.del_bias.shape)
        gradients = self.biasOpt.compute_gradient(self.del_bias)
        self.bias -= gradients  # * learning_rate


class NonLinearLayer(LinearLayer):
    """
        Linear Layer class
        Z = X . W           ; X -> R(num_samples , input_dim)
                            ; W -> R(input_dim , output_dim)
        Y = actv(Z)

        given: dE/dY  (AKA dY)      ; R(num_samples, output_dim)
        dE/dZ = dY * actv'(Z)       ; ''''''''''''''''''''''''''
        dE/dW = dE/dZ . dZ/dW
        .: dW = X.T . dZ    ; R(input_dim, output_dim)
        .: dX = dZ . W.T      ; R(num_samples, input_dim)
        """

    def __init__(self, input_dim, output_dim, activation=None, weights=None, bias=None, optimizer=SGD()):
        if activation is None:
            activation = Linear()
        layerList.pop(len(layerList) - 1)  # removing the activation(last ones) as a seperate layer

        super().__init__(input_dim, output_dim, weights, bias, optimizer)
        layerList.pop(len(layerList) - 1)  # removing the linear_layear(last ones) as a seperate layer

        self.activation = activation
        self.output = None
        self.del_output = None

        layerList.append(self)
        pass

    def forward(self, input):
        super().forward(input)
        self.output = self.activation.forward(self.zee)
        return self.output

    def backward(self, output_delta):
        self.del_output = output_delta
        self.del_zee = self.activation.backward(output_delta)
        return super().backward(self.del_zee)

    def update(self):
        super().update()


class NonLinearLayer_(Layer):  # this class is depricated! will be removed in future versions
    """
        Linear Layer class
        Z = X . W           ; X -> R(num_samples , input_dim)
                            ; W -> R(input_dim , output_dim)
        Y = actv(Z)

        given: dE/dY  (AKA dY)      ; R(num_samples, output_dim)
        dE/dZ = dY * actv'(Z)       ; ''''''''''''''''''''''''''
        dE/dW = dE/dZ . dZ/dW
        .: dW = X.T . dZ    ; R(input_dim, output_dim)
        .: dX = dZ . W.T      ; R(num_samples, input_dim)
        """

    def __init__(self, input_dim, output_dim, activation=None, weights=None, bias=None, optimizer=SGD()):
        if weights is None:
            # self.weights = np.random.randn(input_dim, output_dim) * 2 / (input_dim + output_dim)
            # self.weights = (np.random.randn(input_dim, output_dim) * 0.2) - 0.1
            self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)  # Xavier/He initialization
        else:
            self.weights = weights

        if bias is None:
            self.bias = np.zeros(output_dim)
        else:
            self.bias = bias

        if activation is None:
            activation = Linear()
        layerList.pop(len(layerList) - 1)  # removing the activation(last ones) as a seperate layer

        self.weightsOpt = optimizer.set_parameter(self.weights)
        self.biasOpt = optimizer.set_parameter(self.bias)

        self.activation = activation
        self.input = None
        self.zee = None
        self.output = None

        self.del_weights = None
        self.del_bias = None
        self.del_output = None
        self.del_zee = None

        layerList.append(self)
        pass

    def forward(self, input):
        self.input = input
        # self.zee = input.dot(self.weights) + self.bias
        self.zee = input @ self.weights + self.bias  # @ == .dot
        self.output = self.activation.forward(self.zee)
        return self.output

    def backward(self, output_delta):
        self.del_output = output_delta
        self.del_zee = self.activation.backward(output_delta)
        m = output_delta.shape[0]
        self.del_bias = np.mean(self.del_zee, axis=0)  # * 1 / m
        self.del_weights = self.input.T.dot(self.del_zee) * 1 / m
        # return self.del_output.dot(self.weights.T)
        return self.del_zee.dot(self.weights.T)

    def update(self):
        self.activation.update()
        gradients = self.weightsOpt.compute_gradient(self.del_weights)
        # print(self.weights)
        # print(gradients)
        # print(gradients.shape)
        self.weights -= gradients  # * learning_rate
        # print(self.bias.shape, self.del_bias.shape)
        gradients = self.biasOpt.compute_gradient(self.del_bias)
        self.bias -= gradients  # * learning_rate


class WeightsLayer(Layer):

    def __init__(self, input_dim, output_dim, weights=None, optimizer=SGD()):
        if weights is None:
            # self.weights = np.random.randn(input_dim, output_dim) * 2 / (input_dim + output_dim)
            # self.weights = (np.random.randn(input_dim, output_dim) * 0.2) - 0.1
            self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)  # Xavier/He initialization
        else:
            self.weights = weights

        self.optimizer = optimizer.set_parameter(self.weights)

        self.input = None
        self.output = None

        self.del_weights = None
        self.del_output = None

        layerList.append(self)
        pass

    def forward(self, input):
        self.input = input
        self.output = self.input @ self.weights  # @ == .dot
        return self.output

    def backward(self, output_delta):
        self.del_output = output_delta
        m = output_delta.shape[0]
        self.del_weights = self.input.T.dot(self.del_output) * 1 / m
        return self.del_output.dot(self.weights.T)

    def update(self):
        gradients = self.optimizer.compute_gradient(self.del_weights)
        self.weights -= gradients  # * learning_rate


class BiasLayer(Layer):

    def __init__(self, io_dim, bias=None, optimizer=SGD()):
        if bias is None:
            self.bias = np.zeros(io_dim)
        else:
            self.bias = bias

        self.biasOpt = optimizer.set_parameter(self.bias)

        self.input = None
        self.output = None

        self.del_bias = None
        self.del_output = None

        layerList.append(self)
        pass

    def forward(self, input):
        self.input = input
        self.output = self.input + self.bias  # @ == .dot
        return self.output

    def backward(self, output_delta):
        self.del_output = output_delta
        self.del_bias = np.mean(self.del_output, axis=0)
        return self.del_output

    def update(self):
        gradients = self.biasOpt.compute_gradient(self.del_bias)
        self.bias -= gradients  # * learning_rate


##############################################################
##############################################################


class AutoForm(Layer):

    def __init__(self, new_layers=False):
        self.out = None
        self.input = None
        self.error = None
        self.layerList = []
        global layerList
        if new_layers:
            layerList = self.layerList
        else:
            self.layerList = layerList

    def collect_global_layers(self):
        global layerList
        self.layerList = layerList.copy()
        layerList = []

    def get_layers(self):
        return self.layerList

    def forward(self, input, layers: list = None):
        if layers is not None:
            self.layerList = layers
        self.input = input
        self.out = 0
        for layer in self.layerList:
            # print("layertype",type(layer))
            self.out = layer.forward(input)
            input = self.out
        return self.out

    def calculate_error(self, target, error_type, return_error=False):
        self.error = error_type.del_loss(self.out, target)
        if return_error:
            return error_type.loss(self.out, target)
        return self.error

    def backward(self, error_in=None):
        if error_in is None:
            if self.error is None:
                print('No error available for backpropagation')
        else:
            self.error = error_in
        error_out = 0
        for layer in reversed(self.layerList):
            error_out = layer.backward(self.error)
            self.error = error_out

        self.error = None  # Resetting the error
        return error_out

    def update(self):
        for layer in self.layerList:
            layer.update()

# class LinearLayer(object):
#     """
#     Linear Layer class
#     Y = X . W           ; X -> R(num_samples , input_dim)
#                         ; W -> R(input_dim , output_dim)
#     given: dE/dY        ; R(num_samples, output_dim)
#     dE/dW = dE/dY . dY/dW
#     .: dW = X.T . dY    ; R(input_dim, output_dim)
#     .: dX = dY . W.T      ; R(num_samples, input_dim)
#     """
#
#     def __init__(self, input_dim, output_dim):
#         # self.weights = np.random.randn(input_dim, output_dim) * 2 / (input_dim + output_dim)
#         self.weights = (np.random.randn(input_dim, output_dim) * 0.2) - 0.1
#         self.bias = np.zeros(output_dim)
#
#         self.input = None
#         self.output = None
#         self.del_weights = None
#         self.del_bias = None
#         self.del_output = None
#         pass
#
#     def forward(self, input):
#         self.input = input
#         self.output = input.dot(self.weights) + self.bias
#         return self.output
#
#     def backward(self, output_delta):
#         self.del_output = output_delta
#         self.del_bias = np.mean(self.del_output, axis=0)
#         self.del_weights = self.input.T.dot(self.del_output)
#         return self.del_output.dot(self.weights.T)
#
#     def update(self, learning_rate=0.1):
#         self.weights -= self.del_weights * learning_rate
#         self.bias -= self.del_bias * learning_rate




# class LinearLayerNoBias(Layer):
#     """
#         Linear Layer class
#         Z = X . W           ; X -> R(num_samples , input_dim)
#                             ; W -> R(input_dim , output_dim)
#         Y = actv(Z)
#
#         given: dE/dY  (AKA dY)      ; R(num_samples, output_dim)
#         dE/dZ = dY * actv'(Z)       ; ''''''''''''''''''''''''''
#         dE/dW = dE/dZ . dZ/dW
#         .: dW = X.T . dZ    ; R(input_dim, output_dim)
#         .: dX = dZ . W.T      ; R(num_samples, input_dim)
#         """
#
#     def __init__(self, input_dim, output_dim, activation=Linear(), weights=None, optimizer=SGD()):
#         if weights is None:
#             # self.weights = np.random.randn(input_dim, output_dim) * 2 / (input_dim + output_dim)
#             # self.weights = (np.random.randn(input_dim, output_dim) * 0.2) - 0.1
#             self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)  # Xavier/He initialization
#         else:
#             self.weights = weights
#
#         self.weightsOpt = optimizer.set_parameter(self.weights)
#
#         self.activation = activation
#         self.input = None
#         self.zee = None
#         self.output = None
#
#         self.del_weights = None
#         self.del_output = None
#         self.del_zee = None
#
#         layerList.append(self)
#         pass
#
#     def forward(self, input):
#         self.input = input
#         # self.zee = input.dot(self.weights) + self.bias
#         self.zee = input @ self.weights  # @ == .dot
#         self.output = self.activation.forward(self.zee)
#         return self.output
#
#     def backward(self, output_delta):
#         self.del_output = output_delta
#         self.del_zee = self.del_output * self.activation.backward()
#         m = output_delta.shape[0]
#         self.del_weights = self.input.T.dot(self.del_zee) * 1 / m
#         # return self.del_output.dot(self.weights.T)
#         return self.del_zee.dot(self.weights.T)
#
#     def update(self):
#         gradients = self.weightsOpt.compute_gradient(self.del_weights)
#         self.weights -= gradients  # * learning_rate