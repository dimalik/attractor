import logging

import numpy as np


logger = logging.getLogger(__name__)


class RandomInput(object):
    """Prepares a random binary input dataset of given dimensionality.

    Attributes:
        nb_examples (int): The number of examples to generate. If number
                           is larger than the binary possibilites, all the
                           generated are going to be returned
        nb_dims (int): Number of dimensions
        prob (float \in (0, 1]): The bernoulli probability of each dimension
                                 being set to 1.
        max_iter (int): How many times to search for new possible binary
                        permutations.
    """
    def __init__(self, nb_examples, nb_dims, prob=0.5, max_iter=100):
        self.nb_examples = nb_examples
        self.nb_dims = nb_dims
        self.prob = prob
        self.max_iter = max_iter
        self.matrix = self.getX()

    @staticmethod
    def lexsorting(data):
        """Returns the unique rows in a matrix."""
        sorted_idx = np.lexsort(data.T)
        sorted_data = data[sorted_idx, :]
        row_mask = np.append([True], np.any(np.diff(sorted_data, axis=0), 1))
        return sorted_data[row_mask]

    @staticmethod
    def getExamples(nb_examples, nb_dims, prob):
        return np.array(np.random.binomial(1, prob, size=[nb_examples,
                                                          nb_dims]),
                        dtype='float32')

    def getX(self):
        lexsorted = self.lexsorting(self.getExamples(self.nb_examples,
                                                     self.nb_dims,
                                                     self.prob))
        i = 0
        while lexsorted.shape[0] < self.nb_examples and i < self.max_iter:
            i += 1
            lexsorted_new = self.lexsorting(self.getExamples(self.nb_examples,
                                                             self.nb_dims,
                                                             self.prob))
            lexsorted = self.lexsorting(np.concatenate((lexsorted,
                                                        lexsorted_new),
                                                       axis=0))
        if lexsorted.shape[0] > self.nb_examples:
            return lexsorted[np.random.choice(lexsorted.shape[0],
                                              self.nb_examples,
                                              replace=False), :]
        return lexsorted


class RandomPairedInput(RandomInput, dict):
    """Same as RandomInput with an added dict functionality

    Attributes:
        mappings (iterable): Keys to the dict (values will be the
                             rows from the matrix).

    """
    def __init__(self, mappings, *args, **kwargs):
        self.mappings = mappings
        super(RandomPairedInput, self).__init__(*args, **kwargs)

        for i, word in enumerate(self.mappings):
            self[word] = self.matrix[i, :]


def getRandomInput(nb_examples, nb_dims, prob=0.5, max_iter=100,
                   mappings=None):
    if mappings is not None:
        assert len(mappings) == nb_examples
        return RandomPairedInput(nb_examples=nb_examples, nb_dims=nb_dims,
                                 prob=prob, max_iter=max_iter,
                                 mappings=mappings)
    else:
        return RandomInput(nb_examples=nb_examples, nb_dims=nb_dims,
                           prob=prob, max_iter=max_iter)
    return False
