import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                                    #
    #############################################################################
    N, dim = X.shape
    num_classes = W.shape[1]
    for i in range(N):
        observation_x = X[i, :]
        scores = np.zeros(num_classes) # [K, 1] 
        for cls in range(num_classes):
            w = W[:, cls]
            scores[cls] = w.dot(observation_x)
        # Shift the scores by subtracting max(score)
        scores -= np.max(scores)
        correct_class = y[i]
        sum_exp_scores = np.sum(np.exp(scores))

        corr_cls_exp_score = np.exp(scores[correct_class])
        loss_x = -np.log(corr_cls_exp_score / sum_exp_scores)
        loss += loss_x

        # compute the gradient [D, K]
        percent_exp_score = np.exp(scores) / sum_exp_scores
        for j in range(num_classes):
            dW[:, j] += percent_exp_score[j] * observation_x


        dW[:, correct_class] -= observation_x # deal with the correct class

    loss /= N
    loss += 0.5 * reg * np.sum(W * W) # regularization
    dW /= N
    dW += reg * W
    
    #############################################################################
    #                     END OF YOUR CODE                                      #
    #############################################################################
    

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                                    #
    #############################################################################
    N, dim = X.shape
    scores = X.dot(W) # [K, N]
    # Shift scores by subtracting max(score)
    scores = (scores.T - np.max(scores, axis=1).T).T
    scores_exp = np.exp(scores)
    # get probability scores of observations corresponding to observed class y_i [N, ]
    correct_scores_exp = scores_exp[range(N), y] 
    scores_exp_sum = np.sum(scores_exp, axis=1) # [N, ]
    loss = -np.sum(np.log((correct_scores_exp.T / scores_exp_sum.T).T))
    loss /= N
    loss += 0.5 * reg * np.sum(W * W)

    scores_exp_normalized = (scores_exp.T / scores_exp_sum.T).T
    # deal with the correct class
    scores_exp_normalized[range(N), y] -= 1 # [K, N] this is (p_i,m - Pm)
    dW = X.T.dot(scores_exp_normalized)
    dW /= N
    dW += reg * W
    
    #############################################################################
    #                     END OF YOUR CODE                                      #
    #############################################################################
    

    return loss, dW
