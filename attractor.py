import os
import datetime
import cPickle as pickle
import logging

import numpy as np
from sklearn.base import BaseEstimator

import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid

logger = logging.getLogger(__name__)
mode = theano.Mode(linker='cvm')


class Attractor(object):
    # highest and lowest possible values to be used in clipping
    # numpy just produces nans with lower values
    ZERO = np.float32(np.finfo(theano.config.floatX).resolution)
    ONE = np.float32(1-ZERO)
    
    def __init__(self, input, nin, nout, nticks, tau, activation=sigmoid,
                 output_type='real', **kwargs):
        '''
        kwargs :: W_oi a nout X nin ndarray
                  W_oo a nout X nout ndarray
        '''
        self.nout = nout
        self.nticks = nticks
        self.input = input
        self.activation = activation
        self.output_type = output_type
        self.tau = theano.shared(np.float32(tau), name='tau')
        if "W_oi" not in kwargs:
            logger.info("Input to output weights not provided...Initializing\
by sampling uniformly from -.05 to .05.")
        W_oi = kwargs.get('W_oi',
                          np.array(np.random.uniform(-.05, .05,
                                                     size=(nout, nin)),
                                   dtype=theano.config.floatX))
        # to output from input
        self.W_oi = theano.shared(W_oi, name="W_oi", borrow=True)

        # to output from output
        if "W_oi" not in kwargs:
            logger.info("Output to output weights not provided...Initializing\
by sampling uniformly from -.05 to .05.")
        W_oo = kwargs.get('W_oo',
                          np.array(np.random.uniform(-.05, .05,
                                                     size=(self.nout,
                                                           self.nout)),
                                   dtype=theano.config.floatX))
        # zero the diagonal to avoid self-connections
        W_oo -= np.diag(np.diag(W_oo))
        self.W_oo = theano.shared(W_oo, name="W_oo", borrow=True)
        # output bias
        if "b" not in kwargs:
            logger.info("Bias weights not provided...Initializing to zero.")
        b = kwargs.get("b", np.zeros(nout, dtype=theano.config.floatX))
        self.b = theano.shared(b, name="bias", borrow=True)

        self.params = [self.W_oi, self.W_oo, self.b]

        # for every parameter, we maintain it's last update
        # the idea here is to use "momentum"
        # keep moving mostly in the same direction
        self.upd = {param: theano.shared(
            np.zeros(param.get_value(borrow=True).shape,
                     dtype=theano.config.floatX)) for param in self.params}
        
        if self.output_type == 'real':
            self.loss = lambda y: self.mse(y)
        elif self.output_type == 'binary':
            self.loss = lambda y: self.cross_entropy(y)

    def step(self, out, old, tau, Woi, input, Woo, b):
        # we use clip to avoid 0, 1 which would return
        # nans when computing the cross_entropy error.
        net = (tau * (T.dot(Woi, input) + T.dot(Woo, out) + b)) + (1. - tau) * old
        return [self.activation(net), net]

    def trial(self, init=None):
        if not init:
            init = np.asarray(np.random.uniform(0, .1, size=self.nout),
                              dtype=theano.config.floatX)
        old_activation = np.zeros(self.nout, dtype=theano.config.floatX)
        (results, old), \
            updates = theano.scan(fn=self.step,
                                  outputs_info=[init, old_activation],
                                  non_sequences=[self.tau,
                                                 self.W_oi,
                                                 self.input,
                                                 self.W_oo,
                                                 self.b],
                                  n_steps=self.nticks, strict=True)
        return results
        
    def mse(self, y, n=10):
        return self.tau * T.mean(T.sum(
            (self.trial()[n:] - y) ** 2, axis=1))
    
    def cross_entropy(self, y, n=10):
        '''
        Compute the cross_entropy error between the current input
        and y.

        :parameters:
             y :: theano symbolic variable
                  1d variable containing the target output
             n :: int
                 point after which we average and compute the error
                 zero for the entire matrix
        :returns:
             scalar :: cross_entropy error
        '''        
        return self.tau * T.mean(T.sum(
            T.nnet.binary_crossentropy(self.trial()[n:], y),
            axis=1))


class MetaAttractor(BaseEstimator):
    def __init__(self, nin, nout, epochs, learning_rate, nticks, tau,
                 momentum=1, activation=T.nnet.sigmoid, output_type='real'):
        self.nin = nin
        self.nout = nout
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.nticks = nticks
        self.tau = tau
        self.mom = momentum
        # check for errors
        if hasattr(self.mom, '__iter__') and len(self.mom) != self.epochs:
            raise IndexError('You provided a list of momentum values \
which did not correspond to the number of epochs.')
        self.activation = activation
        self.output_type = output_type
        self.ready()

    def ready(self):
        self.index = T.lscalar('index')
        self.X = T.vector('x', dtype=theano.config.floatX)
        if self.output_type == 'binary':
            self.Y = T.vector('y', dtype='int32')
        elif self.output_type == 'real':
            self.Y = T.vector('y', dtype=theano.config.floatX)
        self.momentum = T.scalar('momentum')
        self.attractor = Attractor(input=self.X, nin=self.nin, nout=self.nout,
                                   nticks=self.nticks, tau=self.tau,
                                   activation=self.activation,
                                   output_type=self.output_type)

    def __getstate__(self):
        """ Return state sequence."""
        params = self.get_params()  # parameters set in constructor
        weights = [p.get_value() for p in self.attractor.params]
        state = (params, weights)
        return state

    def _set_weights(self, weights):
        """ Set fittable parameters from weights sequence.

        Parameters must be in the order defined by self.params:
            W_oi, W_oo, b
        """
        i = iter(weights)

        for param in self.attractor.params:
            param.set_value(i.next())

    def __setstate__(self, state):
        """ Set parameters from state sequence.

        Parameters must be in the order defined by self.params:
            W_oi, W_oo, b
        """
        params, weights = state
        self.set_params(**params)
        self.ready()
        self._set_weights(weights)

    def save(self, fpath='.', fname=None):
        """ Save a pickled representation of Model state. """
        fpathstart, fpathext = os.path.splitext(fpath)
        if fpathext == '.pkl':
            # User supplied an absolute path to a pickle file
            fpath, fname = os.path.split(fpath)

        elif fname is None:
            # Generate filename based on date
            date_obj = datetime.datetime.now()
            date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
            class_name = self.__class__.__name__
            fname = '%s.%s.pkl' % (class_name, date_str)

        fabspath = os.path.join(fpath, fname)

        logger.info("Saving to %s ..." % fabspath)
        file = open(fabspath, 'wb')
        state = self.__getstate__()
        pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()

    def load(self, path):
        """ Load model parameters from path. """
        logger.info("Loading from %s ..." % path)
        file = open(path, 'rb')
        state = pickle.load(file)
        self.__setstate__(state)
        file.close()

    def fit(self, x, y):
        ntrials, ndims = x.shape
        x = theano.shared(x, name='xshared')
        y = theano.shared(y, name='yshared')

        cost = self.attractor.loss(self.Y)

        # this implements Polyak's (1964) classical momentum
        gparams = [T.grad(cost, param) for param in self.attractor.params]
        # alternatively we can use Nesterov's Accelerated Gradient
        # gparams = [T.grad(cost, param + (self.momentum * param))
        #              for param in self.attractor.params]
        # modified update to handle momentum

        updates = {}
        for param, gparam in zip(self.attractor.params, gparams):
            weight_update = self.attractor.upd[param]
            upd = self.momentum * weight_update - self.learning_rate * gparam
            updates[weight_update] = upd
            updates[param] = theano.ifelse.ifelse(
                T.lt(self.momentum, 1),
                param + upd,
                param - self.learning_rate * gparam)

        train_model = theano.function(
            inputs=[self.index, self.momentum],
            outputs=cost,
            updates=updates,
            givens={
                self.X: x[self.index],
                self.Y: y[self.index]})

        for i in range(self.epochs):
            costs = []
            effective_momentum = self.mom[i] if hasattr(self.mom, '__iter__') \
                else self.mom
            for j in range(ntrials):
                costs.append(train_model(j, effective_momentum))
                # avoid self-connections
                cur_value = self.attractor.W_oo.get_value()
                self.attractor.W_oo.set_value(cur_value - np.diag(
                    np.diag(cur_value)))
            print sum(costs) / float(len(costs))

    def activate(self, wordforms, semantic_form, errfun='mse'):
        '''
        :parameters:
             wordforms :: numpy array
                     a |N| x nin ndarray containing the inputs to the network
                     the last activation of a word will be used as the current
                     semantic activation (i.e. it will not reset to the
                     default).
             semantic_form :: numpy array
             measures :: a list

        :returns:
             results :: |N| * nticks x 1|2 ndarray
                 numpy array which contains either the error or the total
                 activation depeding on the parameters given.
        '''

        def cross_entropy(t, o):
            return self.tau * T.sum(T.nnet.binary_crossentropy(o, t), axis=1)

        def mse(t, o):
            return self.tau * T.sum((o - t) ** 2, axis=1)

        fun = mse if errfun == 'mse' else cross_entropy
        
        results = {'errors': [],
                   'semantic_activation': []}
        
        index = T.lscalar('index')

        nexamples = wordforms.shape[0]

        wordforms = theano.shared(wordforms, name="wordforms")
        semantic_form = theano.shared(np.array(semantic_form, dtype='float32'),
                                      name="semantic_form")
        init = theano.shared(np.asarray(np.random.uniform(0, .1,
                                                          size=self.nout),
                                        dtype=theano.config.floatX))
        
        activations = self.attractor.trial(init=init)
        activate = theano.function(inputs=[index],
                                   outputs=activations,
                                   givens={
                                       self.X: wordforms[index]},
                                   updates={init: activations[-1]})
        for i in range(nexamples):
            tmp = activate(i)
            results['errors'].append(fun(semantic_form.get_value()[i],
                                         tmp).eval())
            results['semantic_activation'].append(tmp.sum(axis=1))

        return results


NOUNSPATH = '../data/mcrae_norms/all_nouns.pkl'


epochs = 20
learning_rate = 0.01
ticks = 20
tau = 0.2
momentum = .9
inputdim = 30

with file(NOUNSPATH, "rb") as fin:
    noun_vecs = pickle.load(fin)
noun_vecs = noun_vecs.astype('int32')
ntrials, ndims = noun_vecs.shape


def unique(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return a[ui]

x = np.array(np.random.binomial(1, .5, size=(ntrials,
                                             inputdim)),
             dtype=theano.config.floatX)
assert x.shape == unique(x).shape

myattractor = MetaAttractor(inputdim, ndims, epochs,
                            learning_rate, ticks, tau,
                            momentum, output_type='binary')

myattractor.load("MetaAttractor.2015-11-08-02:14:23.pkl")
x = np.load("x.npy")
# myattractor.fit(x, noun_vecs)
# myattractor.save()
# np.save("x", x)
