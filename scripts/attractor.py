import logging

import numpy as np

import theano
import theano.tensor as T


theano.config.floatX = 'float32'

logger = logging.getLogger(__name__)
theano.config.exception_verbosity = 'high'

MIN = np.finfo('float32').resolution
MAX = (1 - MIN).astype('float32')


class Net(object):

    def cost_function(self):
        raise NotImplementedError

    def gradients(self):
        raise NotImplementedError

    def pre_epoch(self):
        pass

    def post_epoch(self):
        pass

    def save(self):
        pass

    def load(self):
        pass


class AbstractAttractor(Net):
    def cross_entropy(self, y_pred, y_true):
        return T.sum(T.nnet.binary_crossentropy(y_pred, y_true), axis=-1)

    def gradient_updates_momentum(self, cost, params,
                                  learning_rate, momentum):
        assert momentum < 1 and momentum >= 0

        updates = []
        for param in params:
            param_update = theano.shared(param.get_value() * 0.,
                                         broadcastable=param.broadcastable)
            updates.append((param, param - learning_rate * param_update))
            updates.append((param_update, momentum * param_update +
                            (1. - momentum) * T.grad(cost, param)))
        return updates

    def cost_function(self, *args, **kwargs):
        return self.cross_entropy(*args, **kwargs)

    def gradients(self, *args, **kwargs):
        return self.gradient_updates_momentum(*args, **kwargs)


class Attractor(AbstractAttractor):
    def __init__(self, n_in, n_out, n_ticks=20, tau=0.2, train_for=10,
                 activation=T.nnet.sigmoid, batch_size=30):

        self.n_in = n_in
        self.n_out = n_out
        self.n_ticks = n_ticks
        self.tau = tau
        self.activation = activation
        self.train_for = train_for
        self.batch_size = batch_size

        # report vars

        self.input = T.matrix()
        self.y = T.matrix()
        self.init = T.matrix()
        self.old_activation = T.matrix()

        self._init()

    def _init(self):

        logger.info('Initializing weight matrices..')

        logger.info("Initializing input to output weights by sampling \
        uniformly from -.05 to .05.")
        self.W_oi = theano.shared(
            np.array(
                np.random.uniform(-.05, .05,
                                  size=(self.n_in,
                                        self.n_out)),
                dtype=theano.config.floatX),
            name="W_oi",
            borrow=True)

        logger.info("Initializing output to output weights by sampling \
        uniformly from -.05 to .05.")
        W_oo = np.array(np.random.uniform(-.05, .05,
                                          size=(self.n_out,
                                                self.n_out)),
                        dtype=theano.config.floatX)
        logger.info('Removing self-connections')
        W_oo -= np.diag(np.diag(W_oo))  # zero the diagonal
        self.W_oo = theano.shared(W_oo, name="W_oo", borrow=True)

        logger.info("Initializing bias weights to zero.")
        self.b = theano.shared(
            np.zeros(self.n_out,
                     dtype=theano.config.floatX),
            name="bias", borrow=True)

        self.params = [self.W_oi, self.W_oo, self.b]

    def step(self, out, old, tau, Woi, input, Woo, b):
        net = (tau * (T.dot(input, Woi) +
                      T.dot(out, Woo) + b)) + (1. - tau) * old
        act_net = self.activation(net)
        act_net = act_net.clip(MIN, MAX)  # avoid nans
        return [act_net, net]

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
        return [results, old]

    def post_epoch(self):
        cur_value = self.W_oo.get_value()
        self.W_oo.set_value(cur_value - np.diag(
            np.diag(cur_value)))

    def fit(self, X_, y_, init=None, learning_rate=0.1, momentum=0.9,
            nb_epochs=20):

        i = T.lscalar()

        if init is None:
            init = np.array(np.random.uniform(0, .01,
                                              size=(self.batch_size,
                                                    self.n_out)),
                            dtype='float32')

        old_activation = np.zeros((self.batch_size, self.n_out),
                                  dtype=theano.config.floatX)

        cost = self.tau * (
            self.cost_function(
                self.trial()[0][self.train_for:],
                self.y).sum() / self.train_for)

        train = theano.function(
            [self.input, self.y],
            cost,
            updates=self.gradients(cost,
                                   self.params,
                                   learning_rate,
                                   momentum),
            givens={self.init: init,
                    self.old_activation: old_activation})

        for i in range(nb_epochs):
            self.pre_epoch()
            epoch_cost = train(X_, y_)
            print epoch_cost
            logger.info('Epoch: %d\tCost: %f' % (i + 1, epoch_cost))
            self.post_epoch()

    def get_priming(self, word_vecs):
        init = np.array(np.random.uniform(0, .01,
                                          size=(1, self.n_out)),
                        dtype='float32')

        old_activation = np.array(np.random.uniform(0, .01,
                                                    size=(1, self.n_out)),
                                  dtype='float32')

        # old_activation = np.zeros((1, self.n_out),
        #                           dtype=theano.config.floatX)

        fun = theano.function([self.input, self.init, self.old_activation],
                              self.trial())
        reslist = []

        for word in word_vecs:
            results, old = fun(np.array(word, ndmin=2), init, old_activation)
            init = np.array(results[-1], ndmin=2)
            old_activation = np.array(old[-1], ndmin=2)
            reslist.append(results)
        return np.concatenate(reslist)

    # def get_activation(self, word_vecs):
    #     fun = theano.function([self.input, ])
