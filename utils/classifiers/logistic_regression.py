import numpy as np
from random import shuffle

def sigmoid(x):
    h = np.zeros_like(x)
    
    #############################################################################
    # TODO: Implement sigmoid function.                                         #         
    #############################################################################
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################
    h = 1 / (1 + np.exp(-x))
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    
    
    return h 

def logistic_regression_loss_naive(W, X, y, reg):
    """
      Logistic regression loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c can be either 0 or 1.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the logistic regression loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    num_train, dim = X.shape
    num_classes = W.shape[1]
    h_x = np.zeros((num_train, num_classes))
    y_final = np.column_stack((y, 1 - y))
    for i in range(num_train):
        sample_x = X[i, :]
        sample_x=sample_x
        scores = np.zeros(num_classes) # [K, 1] unnormalized score
        for cls in range(num_classes):
            w = W[:, cls]
            w=w.reshape((dim, 1))
            scores[cls] = sample_x.reshape((1, dim)).dot(w)   
        h_x[i,:] = sigmoid(scores)
        correct_class = y_final[i,:]
        loss_x = correct_class * np.log(h_x[i, :]) + (1 - correct_class) * np.log(1 - h_x[i, :])
        loss += loss_x
        loss = loss[0]
    
    # compute the gradient #[D, N] * [N, C] = [D, C]
        for j in range(num_classes):
            dW[:, j] += (h_x[i, j]-y_final[i,j]) * sample_x #[N, 1] * [1, D] = [D, 1]


    #grad[correct_class, :] -= sample_x # deal with the correct class
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W) # add regularization
    loss = -loss
    dW /= num_train
    dW += reg * W
    

    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################
    

    return loss, dW



def logistic_regression_loss_vectorized(W, X, y, reg):
    """
    Logistic regression loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the logistic regression loss and its gradient using no     # 
    # explicit loops.                                                          #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the       #
    # regularization!                                                          #
    ############################################################################
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################
    num_train, dim = X.shape
    num_classes = W.shape[1]
    y_final = np.column_stack((y, 1 - y))
    f_x_mat = X.dot(W) # [N, D] x [D, C] matrices
    h_x_mat = 1.0 / (1.0 + np.exp(-f_x_mat)) # [N, C]
    h_x = h_x_mat[:, 0] # get probability scores of first class (y=1)
    loss = np.sum(y * np.log(h_x) + (1 - y) * np.log(1 - h_x))
    loss = -1.0 * loss / num_train  + 0.5 * reg * np.sum(W * W)
    
    
    y_final = np.column_stack((y, 1 - y)) #get y_i's (corresponding label to x_i)
    dW = X.T.dot(h_x_mat - y_final) # [K, D]
    dW = dW / (num_train) + reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    

    return loss, dW