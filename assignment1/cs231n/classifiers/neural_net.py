# Add nesterov momentum and dropout, compared to original version

from __future__ import print_function
from six.moves import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.softmax import *
from past.builtins import xrange

def relu(x): return np.maximum(0., x)

def relu_prime(x):
  derivative = np.maximum(0, x)
  derivative[derivative > 0] = 1
  return derivative

def dropout_layer(layer_output, p_dropout):
    mask = np.random.binomial(n=1, p=1-p_dropout, size=layer_output.shape)
    mask = mask.astype('float32')
    return layer_output * mask

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, dropout_rate=0., std=1e-3,
               init_mode='normal'):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    - dropout_rate: Dropout rate of the hidden layer.
    """
    self.params = {}
    self.params['b1'] = np.zeros(hidden_size)
    self.params['b2'] = np.zeros(output_size)
    if init_mode == 'normal':
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    elif init_mode == 'deep':
        self.params['W1'] = np.sqrt(2/input_size) * np.random.randn(input_size, hidden_size)
        self.params['W2'] = np.sqrt(1/hidden_size) * np.random.randn(hidden_size, output_size)
    self.dropout_rate = dropout_rate

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    # If the targets are not given, we're at testing phase
    if y is None:
      a1 = relu((1-self.dropout_rate) * X.dot(W1) + b1)
      scores = softmax((1-self.dropout_rate) * a1.dot(W2) + b2, 1)
      return scores

    # Training phase
    X_dropout = dropout_layer(X, self.dropout_rate)
    z1 = X_dropout.dot(W1) + b1
    z1 = dropout_layer(z1, self.dropout_rate)
    a1 = relu(z1)
    z2 = a1.dot(W2) + b2
    scores = softmax(z2, 1)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Compute the loss
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    loss = -np.mean(np.log(scores[np.arange(N), y] + 1e-8))
    loss += reg * sum([np.sum(W1 * W1), np.sum(W2 * W2)])
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    delta = scores
    delta[np.arange(N), y] -= 1
    grads['W2'] = a1.T.dot(delta)
    grads['b2'] = delta.sum(0)
    delta = delta.dot(W2.T) * relu_prime(z1)
    grads['W1'] = X_dropout.T.dot(delta)
    grads['b1'] = delta.sum(0)

    grads['W1'] = grads['W1']/N + (2 * reg * W1)
    grads['b1'] = grads['b1']/N
    grads['W2'] = grads['W2']/N + (2 * reg * W2)
    grads['b2'] = grads['b2']/N
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100, optimizer='sgd',
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_loss = []
    val_loss = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.

    moments = {w:np.zeros_like(self.params[w]) for w in self.params}
    for it in xrange(num_iters):
      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      start_idx = int(it % iterations_per_epoch)
      X_batch = X[start_idx*batch_size : start_idx*batch_size+batch_size]
      y_batch = y[start_idx*batch_size : start_idx*batch_size+batch_size]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      # SGD
      if optimizer == 'sgd':
        self.params = {w:self.params[w] - learning_rate*grads[w] for w in self.params}
      # Momentum:
      else:
        velocity = {w: 0.9*moments[w]-learning_rate*grads[w] for w in self.params}
        moments = {w:velocity[w] for w in self.params}
        if optimizer == 'momentum':
          self.params = {w:self.params[w] + velocity[w] for w in self.params}
        # Nesterov momentum
        elif optimizer == 'nag':
          self.params = {w:self.params[w] + 0.9*velocity[w] - learning_rate*grads[w] \
                         for w in self.params}
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val loss/accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        train_loss.append(loss)
        val_loss.append(self.loss(X_val, y_val, reg)[0])
        train_accs.append((self.predict(X_batch) == y_batch).mean())
        val_acc = (self.predict(X_val) == y_val).mean()
        val_accs.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

        # Shuffle the training data / save the best net using val acc
        if it > 0:
          shuffle = np.random.permutation(num_train)
          X, y = X[shuffle], y[shuffle]
          if val_acc > best_val_acc:
            print('Saving the best model so far...')
            with open('./best_net.pkl', 'wb') as f:
              pickle.dump(self, f)

    # Check if the model gets better finally
    val_acc = (self.predict(X_val) == y_val).mean()
    if val_acc > best_val_acc:
      print('Saving the best model so far...')
      with open('./best_net.pkl', 'wb') as f:
        pickle.dump(self, f)

    return {
      'loss_history': loss_history,
      'train_loss': train_loss,
      'val_loss': val_loss,
      'train_acc': train_accs,
      'val_acc': val_accs,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    scores = self.loss(X)
    y_pred = np.argmax(scores, 1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred

