import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax(z, axis=None):
  # shift for fixing numerical instability
  if axis:
    shifted_z = z - np.expand_dims(np.max(z, axis), axis)
    return np.exp(shifted_z) / np.expand_dims(np.sum(np.exp(shifted_z), axis), axis)
  else:
    shifted_z = z - np.max(z)
    return np.exp(shifted_z) / np.sum(np.exp(shifted_z))


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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    scores = softmax(X[i].dot(W))
    loss -= np.log(scores[y[i]])
    scores[y[i]] -= 1                           # to calculate the gradient, with length (C, )
    dW = dW + X[i][:,None].dot(scores[None,:])  # (D, 1) dot (1, C), with shape (D, C)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(W * W)
  dW += 2*reg*W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = softmax(X.dot(W), 1)         # score matrix, with shape (N, C)
  loss -= np.mean(np.log(scores[np.arange(num_train), y]))

  scores[np.arange(num_train), y] -= 1  # to calculate the gradient
  dW = dW + X.T.dot(scores)             # X^T dot softmax(X dot W), with shape (D, C)
  dW /= num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss += reg * np.sum(W * W)
  dW += 2*reg*W

  return loss, dW

