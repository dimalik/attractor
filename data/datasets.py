import cPickle as pkl

import numpy as np
import pandas as pd

from scripts import McRaeModel


def pickleLoader(path):
    with file(path, 'rb') as fin:
        tmp = pkl.load(fin) 
    return tmp


class RandomInput(object):
    def __init__(self, nb_examples, nb_dims, prob=0.5, max_iter=100):
        self.nb_examples = nb_examples
        self.nb_dims = nb_dims
        self.prob = prob
        self.max_iter = max_iter
        self.matrix = self.getX()

    @staticmethod
    def lexsorting(data):
        sorted_idx = np.lexsort(data.T)
        sorted_data =  data[sorted_idx,:]
        row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0), 1))
        return sorted_data[row_mask]

    @staticmethod
    def getExamples(nb_examples, nb_dims, prob):
        return np.random.binomial(1, prob, size=[nb_examples, nb_dims])

    def getX(self):
        lexsorted = self.lexsorting(self.getExamples(self.nb_examples,
                                                     self.nb_dims,
                                                     self.prob))
        i = 0
        while lexsorted.shape[0] < self._nb_examples and i < self.max_iter:
            i += 1
            lexsorted_new = self.lexsorting(self.getExamples(self.nb_examples,
                                                             self.nb_dims,
                                                             self.prob))
            lexsorted = self.lexsorting(np.concatenate((lexsorted,
                                                        lexsorted_new),
                                                       axis=0))
        if lexsorted.shape[0] > self.nb_examples:
            return lexsorted[np.random.choice(lexsorted.shape[0],
                                              self.nb_examples, replace=False), :]
        return lexsorted


class Experiment(object):
    def __init__(self, mcrae_path, words_path):
        self.mcrae_path = mcrae_path
        self.words_path = words_path

    def initModel(self):
        mcrae_all = McRaeModel()
        mcrae_all.load(self.mcrae_path)

        # McRae & Boisvert, 1998
        self.words = pickleLoader(self.words_path)
        self.include = []
        self.exclude = []

        for word in set([x for y in self.words.values() for x in y]):
            if word in mcrae_all.dicts[0]:
                self.include.append(word)
            else:
                self.exclude.append(word)

        self.experiment = mcrae_all.getWordMatrix(*self.include)

    def prepDataset(self):
        raise NotImplementedError
        

class McRaeBoisvertExperiment(Experiment):

    def prepDataset(self):
        primes = self.words['Similar'] + self.words['Dissimilar']
        targets = self.words['Target'] * 2
        desc = ['Similar'] * len(self.words['Target']) + \
               ['Dissimilar'] * len(self.words['Target'])
        
        self.df = pd.DataFrame(data={'Primes': primes, 'Targets': targets,
                                'Description': desc})

        self.df.drop(self.df.loc[self.df.isin(self.exclude).any(1)].index,
                     inplace=True)
