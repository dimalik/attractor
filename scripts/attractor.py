import logging

import numpy as np

import theano
import theano.tensor as T

logger = logging.getLogger(__name__)


class AbstractNet(object):
    def cross_entropy(self, y_pred, y_true):
        return T.sum(T.nnet.binary_crossentropy(y_pred, y_true), axis=-1)

    def gradient_updates_momentum(self, cost, params, learning_rate, momentum):
        assert momentum < 1 and momentum >= 0

        updates = []
        for param in params:
            param_update = theano.shared(param.get_value() * 0.,
                                         broadcastable=param.broadcastable)
            updates.append((param, param - learning_rate * param_update))
            updates.append((param_update, momentum * param_update +
                            (1. - momentum) * T.grad(cost, param)))
        return updates


class Attractor(AbstractNet):
    def __init__(self, n_in, n_out, n_ticks=20, tau=0.2,
                 activation=T.nnet.sigmoid, train_for=10):

        self.n_in = n_in
        self.n_out = n_out
        self.n_ticks = n_ticks
        self.tau = tau
        self.activation = activation
        self.train_for = train_for

        self.input = T.matrix()
        self.init = T.matrix()
        self.old_activation = T.matrix()

        logger.info("Input to output weights not provided...Initializing\
by sampling uniformly from -.05 to .05.")
        self.W_oi = theano.shared(
            np.array(np.random.uniform(-.05, .05,
                                       size=(self.n_in, self.n_out)),
                     dtype=theano.config.floatX), name="W_oi", borrow=True)

        logger.info("Output to output weights not provided...Initializing\
by sampling uniformly from -.05 to .05.")
        W_oo = np.array(np.random.uniform(-.05, .05,
                                          size=(self.n_out,
                                                self.n_out)),
                        dtype=theano.config.floatX)

        W_oo -= np.diag(np.diag(W_oo))
        self.W_oo = theano.shared(W_oo, name="W_oo", borrow=True)

        logger.info("Bias weights not provided...Initializing to zero.")
        self.b = theano.shared(
            np.zeros(n_out,
                     dtype=theano.config.floatX),
            name="bias", borrow=True)

        self.params = [self.W_oi, self.W_oo, self.b]

    def step(self, out, old, tau, Woi, input, Woo, b):
        net = (tau * (T.dot(input, Woi) +
                      T.dot(out, Woo) + b)) + (1. - tau) * old
        return [self.activation(net), net]

    def trial(self):
        (results, old), \
            updates = theano.scan(fn=self.step,
                                  outputs_info=[self.init,
                                                self.old_activation],
                                  non_sequences=[self.tau,
                                                 self.W_oi,
                                                 self.input,
                                                 self.W_oo,
                                                 self.b],
                                  n_steps=self.n_ticks, strict=True)
        return results[self.train_for:]

    def fit(self, X, y_, init=None, learning_rate=0.1, momentum=0.9,
            batch_size=50, nb_epochs=50):

        y = T.matrix()
        old_activation = np.zeros((batch_size, self.n_out),
                                  dtype=theano.config.floatX)

        if init is None:
            init = np.random.uniform(0, .01, size=(batch_size, self.n_out))

        cost = self.tau * (
            self.cross_entropy(self.trial(), y).sum() / self.train_for)

        train = theano.function(
            [self.input, y],
            cost,
            updates=self.gradient_updates_momentum(cost,
                                                   self.params,
                                                   learning_rate,
                                                   momentum),
            givens={self.init: init,
                    self.old_activation: old_activation})

        for i_ in range(nb_epochs):
            print train(X, y_)
            cur_value = self.W_oo.get_value()
            self.W_oo.set_value(cur_value - np.diag(
                np.diag(cur_value)))
