from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange
import math


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_trains = X.shape[0]
    num_classes = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(num_trains):
        scores = X[i].dot(W)
        if i == 0:
            print(scores[y[i]])
            print(np.sum(np.exp(scores)))
            print(-1 * math.log(np.exp(scores[y[i]])/np.sum(np.exp(scores))))
        loss += -1 * math.log(np.exp(scores[y[i]])/np.sum(np.exp(scores)))
        p = np.zeros(num_classes)
        p[0:y[i]] = np.exp(scores[0:y[i]])/np.sum(np.exp(scores))
        p[y[i]:] = np.exp(scores[y[i]:])/np.sum(np.exp(scores))
        p[y[i]] = -1 + np.exp(scores[y[i]])/np.sum(np.exp(scores))
        dW += np.dot(X[i].reshape(-1, 1), p.reshape(1, -1))
    loss /= num_trains
    loss += reg * np.sum(W * W)  
    dW /= num_trains
    dW += 2 * reg * W                                          
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_trains = X.shape[0]
    num_classes = W.shape[1] 
    scores = X.dot(W)
    loss = np.sum(-1 * np.log(np.exp(scores[np.arange(num_trains), y])/np.sum(np.exp(scores), axis = 1)))
    p = np.zeros((num_trains, num_classes))
    p = np.exp(scores)/np.sum(np.exp(scores), axis = 1).reshape(-1, 1)
    p[np.arange(num_trains), y] -= 1
    loss /= num_trains
    loss += reg * np.sum(W * W)    
    dW = np.dot(np.transpose(X), p)
    dW /= num_trains
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
