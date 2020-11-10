from __future__ import print_function

import numpy as np

from utils.layer_funcs import *
from utils.layer_utils import *

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:
    input -> DenseLayer -> AffineLayer -> softmax loss -> output
    Or more detailed,
    input -> affine transform -> ReLU -> affine transform -> softmax -> output

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_dim=3072, hidden_dim=200, num_classes=10, reg=0.0, weight_scale=1e-3):
        """
        Inputs:
        - reg: (float) L2 regularization
        - weight_scale: (float) for layer weight initialization
        """
        self.layer1 = DenseLayer(input_dim, hidden_dim, weight_scale=weight_scale)
        self.layer2 = AffineLayer(hidden_dim, num_classes, weight_scale=weight_scale)
        self.reg = reg
        self.velocities = None

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
        ###################################################
        #            START OF YOUR CODE                   #
        ################################################### 
        ###################################################
        # TODO: Feedforward                               #
        ###################################################
        layer1 = self.layer1
        layer2 = self.layer2
        
        output_layer1 = layer1.feedforward(X)
        output_layer2 = layer2.feedforward(output_layer1)
        loss, dy = softmax_loss(output_layer2, y)
        
        ###################################################
        # TODO: Backpropogation,here is just one dense    #
        # layer, it should be pretty easy                 #
        ###################################################      
        dout = dy
        dX2 = layer2.backward(dout)
        dX1 = layer1.backward(dX2)
        
        #layer1.backward(dout)
        
        ###################################################
        #            END OF YOUR CODE                     #
        ###################################################
        
        # Add L2 regularization
        square_weights = np.sum(layer1.params[0]**2) + np.sum(layer2.params[0]**2)
        loss += 0.5*self.reg*square_weights
        return loss

    def step(self, learning_rate=1e-5, optim='SGD', momentum=0.5):
        """
        Use SGD to implement a single-step update to each weight and bias.
        """
        # creates new lists with all parameters and gradients
        layer1, layer2 = self.layer1, self.layer2
        params = layer1.params + layer2.params
        grads = layer1.gradients + layer2.gradients
        
        if self.velocities is None:
            self.velocities = [np.zeros_like(param) for param in params]
        
        # Add L2 regularization
        reg = self.reg
        grads = [grad + reg*params[i] for i, grad in enumerate(grads)]
        ###################################################
        # TODO: Use SGD or SGD with momentum to update    #
        # variables in layer1 and layer2.                 #
        ###################################################
        ###################################################
        #            START OF YOUR CODE                   #
        ###################################################  
#        # layer 2 - update b
#        params[3] = params[3] - (learning_rate * grads[3])
#        
#        # layer 2 - update 2
#        params[2] = params[2] - (learning_rate * grads[2])
#        
#        # layer 1 - update b
#        params[1] = params[1] - (learning_rate * grads[1])
#        
#        # layer 1 - update w
#        params[0] = params[0] - (learning_rate * grads[0])
        
        
        
        # SGD With Momentum v = momentum*v + learning_rate * gradient
        # layer 2 - update b
        velocities = self.velocities
        
        velocities[3] = momentum*velocities[2] + learning_rate * grads[3]
        params[3] = params[3] - (velocities[3])
        
        # layer 2 - update w
        velocities[2] = momentum*velocities[1] + learning_rate * grads[2]
        params[2] = params[2] - (velocities[2])
        
        # layer 1 - update b
        velocities[1] = momentum*velocities[0] + learning_rate * grads[1]
        params[1] = params[1] - (velocities[1])
        
        # layer 1 - update w
        velocities[0] = momentum*velocities[0] + learning_rate * grads[0]
        params[0] = params[0] - (velocities[0])
        #layer1.w = layer2.w - learning_rate * grads
        
        ###################################################
        #            END aOF YOUR CODE                     #
        ###################################################
        
   
        # update parameters in layers
        layer1.update_layer(params[0:2])
        layer2.update_layer(params[2:4])
        
        # Python Reference, actually there is no need for this step.
        # Here add these two lines to make code more clear.
        self.layer1 = layer1
        self.layer2 = layer2


    def predict(self, X):
        """
        Return the label prediction of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        
        Returns: 
        - predictions: (int) an array of length N
        """
        predictions = None
        layer1, layer2 = self.layer1, self.layer2
        ###################################################
        # TODO: Remember to use functions in class        #
        # SoftmaxLayer.                                   #   
        ###################################################
        ###################################################
        #            START OF YOUR CODE                   # 
        ###################################################  
        #argmax(softmax(X))
        output_layer1 = layer1.feedforward(X)
        output_layer2 = layer2.feedforward(output_layer1)
        scores = output_layer2 - np.max(output_layer2)
        scores_exp = np.exp(scores)
        scores_exp_sum = np.sum(scores_exp, axis=0)
        scores_exp_normalized = scores_exp / scores_exp_sum
        predictions = np.argmax(scores_exp_normalized, axis=1)
        
        ###################################################
        #            END OF YOUR CODE                     #
        ###################################################
        
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
    
    def save_model(self):
        """
        Save model's parameters, including two layer's W and b and reg
        """
        return [self.layer1.params, self.layer2.params, self.reg]
    
    def update_model(self, new_params):
        """
        Update layers and reg with new parameters
        """
        layer1_params, layer2_params, reg = new_params
        
        self.layer1.update_layer(layer1_params)
        self.layer2.update_layer(layer2_params)
        self.reg = reg

        
        