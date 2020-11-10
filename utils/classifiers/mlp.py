from builtins import range
from builtins import object
import numpy as np

from utils.layer_funcs import *
from utils.layer_utils import *

class MLP(object):
    """
    MLP with an arbitrary number of dense hidden layers,
    and a softmax loss function. For a network with L layers,
    the architecture will be

    input >> DenseLayer x (L - 1) >> AffineLayer >> softmax_loss >> output

    Here "x (L - 1)" indicate to repeat L - 1 times. 
    """
    def __init__(self, input_dim=3072, hidden_dims=[200,200], num_classes=10, reg=0.0, weight_scale=1e-3):
        """
        Inputs:
        - reg: (float) L2 regularization
        - weight_scale: (float) for layer weight initialization
        """
        self.num_layers = len(hidden_dims) + 1
        self.reg = reg
        
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims)-1):
            layers.append(DenseLayer(input_dim=dims[i], output_dim=dims[i+1], weight_scale=weight_scale))
        layers.append(AffineLayer(input_dim=dims[-1], output_dim=num_classes, weight_scale=weight_scale))
        
        self.layers = layers

    def loss(self, X, y):
        """
        Calculate the cross-entropy loss and then use backpropogation
        to get gradients wst W,b in each layer.
        
        Inputs:
        - X: input data
        - y: ground truth
        
        Return loss value(float)
        """
        loss = 0.0
        reg = self.reg
        num_layers = self.num_layers
        layers = self.layers
        
        ####################################################
        #           START OF YOUR CODE           #
        ####################################################
        ####################################################
        # TODO: Feedforward                      #
        ####################################################
        output = []
        output.append(layers[0].feedforward(X))
        for i in range(1,num_layers):
            out = layers[i].feedforward(output[i-1])
            output.append(out)
            
        loss, dy = softmax_loss(output[num_layers-1], y)
            
        ####################################################
        # TODO: Backpropogation                   #
        ####################################################
        dout = {}
        dout["dX"+str(num_layers)] = dy
        for i in range(num_layers):
            dout["dX"+str(num_layers-i-1)] = layers[num_layers-i-1].backward(dout["dX"+str(num_layers-i)])

        
#         dX3 = dy
#         dX2 = layer2.backward(dX3)
#         dX1 = layer1.backward(dX2)
#         dX0 = layer0.backward(dx1)
        ####################################################
        # TODO: Add L2 regularization               #
        ####################################################
        square_weights = 0
        for i in range(num_layers):
            square_weights+=np.sum(layers[i].params[0]**2) 
        
        loss += 0.5*self.reg*square_weights
        
        ####################################################
        #            END OF YOUR CODE            #
        ####################################################
        
        
        return loss

    def step(self, learning_rate=1e-5):
        """
        Use SGD to implement a single-step update to each weight and bias.
        """
        ####################################################
        # TODO: Use SGD to update variables in layers.     #
        ####################################################
        ####################################################
        #           START OF YOUR CODE                     #
        ####################################################
        layers = self.layers
        num_layers = self.num_layers
        
        params = [] # create list of params
        for i in range(num_layers):
            params.append(layers[i].params[0])
            params.append(layers[i].params[1])
        
        grads = [] # create list of gradients
        for i in range(num_layers):
            grads.append(layers[i].gradients[0])
            grads.append(layers[i].gradients[1])
        
        
        for i in range(2*(num_layers)): # update gradients
            params[i] = params[i] - learning_rate * grads[i]
            
        
        ####################################################
        #            END OF YOUR CODE                      #
        ####################################################
        
   
        # update parameters in layers
        for i in range(num_layers):
            self.layers[i].update_layer(params[2*i:2*(i+1)])
        

    def predict(self, X):
        """
        Return the label prediction of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        
        Returns: 
        - predictions: (int) an array of length N
        """
        predictions = None
        num_layers = self.num_layers
        layers = self.layers
        #####################################################
        # TODO: Remember to use functions in class          #
        # SoftmaxLayer                                      #
        #####################################################
        ####################################################
        #           START OF YOUR CODE                     #
        ####################################################
        output = []
        output.append(layers[0].feedforward(X))
        for i in range(1,num_layers):
            out = layers[i].feedforward(output[i-1])
            output.append(out)
        
        predictions = np.argmax(output[num_layers-1], axis=1)
        
        ####################################################
        #            END OF YOUR CODE                      #
        ####################################################
        
        return predictions
    
    def check_accuracy(self, X, y):
        """
        Return the classification accuracy of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        - y: (int) an array of length N. ground truth label 
        Returns: 
        - acc: (float) between 0 and 1
        """
        y_pred = self.predict(X)
        acc = np.mean(np.equal(y, y_pred))
        
        return acc
        
        
