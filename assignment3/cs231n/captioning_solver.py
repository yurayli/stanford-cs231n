from __future__ import print_function, division
from builtins import range
from builtins import object
import pickle
import numpy as np
import nltk

from cs231n import optim
from cs231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions


class CaptioningSolver(object):
    """
    A CaptioningSolver encapsulates all the logic necessary for training
    image captioning models. The CaptioningSolver performs stochastic gradient
    descent using different update rules defined in optim.py.

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    To train a model, you will first construct a CaptioningSolver instance,
    passing the model, dataset, and various options (learning rate, batch size,
    etc) to the constructor. You will then call the train() method to run the
    optimization procedure and train the model.

    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_bleu_history and solver.val_bleu_history will be lists containing
    the bleu scores of the model on the training and validation set at each epoch.

    Example usage might look something like this:

    data = load_coco_data()
    model = MyAwesomeModel(hidden_dim=100)
    solver = CaptioningSolver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()


    A CaptioningSolver works on a model object that must conform to the following
    API:

    - model.params must be a dictionary mapping string parameter names to numpy
      arrays containing parameter values.

    - model.loss(features, captions) must be a function that computes
      training-time loss and gradients, with the following inputs and outputs:

      Inputs:
      - features: Array giving a minibatch of features for images, of shape (N, D
      - captions: Array of captions for those images, of shape (N, T) where
        each element is in the range (0, V].

      Returns:
      - loss: Scalar giving the loss
      - grads: Dictionary with the same keys as self.params mapping parameter
        names to gradients of the loss with respect to those parameters.
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new CaptioningSolver instance.

        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data from load_coco_data

        Optional arguments:
        - update_rule: A string giving the name of an update rule in optim.py.
          Default is 'sgd'.
        - optim_config: A dictionary containing hyperparameters that will be
          passed to the chosen update rule. Each update rule requires different
          hyperparameters (see optim.py) but all update rules require a
          'learning_rate' parameter so that should always be present.
        - lr_decay: A scalar for learning rate decay; after each epoch the learning
          rate is multiplied by this value.
        - batch_size: Size of minibatches used to compute loss and gradient during
          training.
        - num_epochs: The number of epochs to run for during training.
        - print_every: Integer; training losses will be printed every print_every
          iterations.
        - verbose: Boolean; if set to false then no output will be printed during
          training.
        """
        self.model = model
        self.data = data

        # Unpack keyword arguments
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.step_size = kwargs.pop('step_size', 10)
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.num_train_samples = kwargs.pop('num_train_samples', 1000)
        self.num_val_samples = kwargs.pop('num_val_samples', 1000)

        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()


    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_bleu = 0
        self.best_params = {}
        self.loss_history = []
        self.val_loss_history = []
        self.train_bleu_history = []
        self.val_bleu_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d


    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Make a minibatch of training data
        minibatch = sample_coco_minibatch(self.data,
                      batch_size=self.batch_size,
                      split='train')
        captions, features, urls = minibatch

        # Compute loss and gradient
        loss, grads = self.model.loss(features, captions)

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

        return loss


    def _save_checkpoint(self):
        if self.checkpoint_name is None: return
        checkpoint = {
          'model': self.model,
          'update_rule': self.update_rule,
          'lr_decay': self.lr_decay,
          'optim_config': self.optim_config,
          'batch_size': self.batch_size,
          'num_train_samples': self.num_train_samples,
          'num_val_samples': self.num_val_samples,
          'epoch': self.epoch,
          'loss_history': self.loss_history,
          'val_loss_history': self.val_loss_history,
          'train_bleu_history': self.train_bleu_history,
          'val_bleu_history': self.val_bleu_history,
        }
        filename = '%s.pkl' % (self.checkpoint_name, )
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)


    def check_bleu(self, split, num_samples, batch_size=100, check_loss=False):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        - split: String 'train' or 'val'
        - num_samples: Subsample the data and only test the model on num_samples
          datapoints.
        - batch_size: Split data into batches of this size to avoid using too
          much memory.

        Returns:
        - bleu: Scalar giving the words that were correctly generated by the model.
        """

        # Subsample the data
        minibatch = sample_coco_minibatch(self.data,
                      batch_size=num_samples,
                      split=split)
        captions, features, urls = minibatch
        if check_loss: loss, _ = self.model.loss(features, captions)
        captions = decode_captions(captions, self.data['idx_to_word'])
        N = num_samples

        # Compute word generations in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        total_score = 0
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            sample_captions = self.model.sample(features[start:end])
            sample_captions = decode_captions(sample_captions, self.data['idx_to_word'])
            for gt_caption, sample_caption in zip(captions, sample_captions):
                total_score += BLEU_score(gt_caption, sample_caption)

        if check_loss:
            return loss, total_score / N

        return total_score / N


    def train(self):
        """
        Run optimization to train the model.
        """
        num_train = self.data['train_captions'].shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            loss = self._step()

            # Maybe print training loss
            if self.verbose and (t + 1) % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (t + 1, num_iterations, loss))

            # At the end of every epoch, increment the epoch counter and decay the
            # learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                if self.epoch % self.step_size == 0:
                    for k in self.optim_configs:
                        self.optim_configs[k]['learning_rate'] *= self.lr_decay

            # Check train and val bleu scores on the first iteration, the last
            # iteration, and at the end of each epoch.
            first_it = (t == 0)
            last_it = (t == num_iterations - 1)
            if first_it or last_it or epoch_end:
                train_bleu = self.check_bleu('train', num_samples=self.num_train_samples)
                val_loss, val_bleu = self.check_bleu('val',
                                        num_samples=self.num_val_samples, check_loss=True)
                self.loss_history.append(loss)
                self.val_loss_history.append(val_loss)
                self.train_bleu_history.append(train_bleu)
                self.val_bleu_history.append(val_bleu)

                if self.verbose:
                    print('(Epoch %d / %d) train loss: %f; val loss: %f' % (
                           self.epoch, self.num_epochs, loss, val_loss))
                    print('; with train bleu: %f; val bleu: %f' % (train_bleu, val_bleu))

                # Keep track of the best model
                if val_bleu > self.best_val_bleu:
                    self._save_checkpoint()
                    self.best_val_bleu = val_bleu
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

                print()

        # At the end of training swap the best params into the model
        self.model.params = self.best_params


def BLEU_score(gt_caption, sample_caption):
    """
    gt_caption: string, ground-truth caption
    sample_caption: string, your model's predicted caption
    Returns unigram BLEU score.
    """
    reference = [x for x in gt_caption.split(' ')
                 if ('<END>' not in x and '<START>' not in x and '<UNK>' not in x)]
    hypothesis = [x for x in sample_caption.split(' ')
                  if ('<END>' not in x and '<START>' not in x and '<UNK>' not in x)]
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights = [1])
    return BLEUscore