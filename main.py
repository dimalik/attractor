import os
import logging

import numpy as np
import matplotlib.pyplot as plt

from config import options
from scripts import Attractor, RandomInput, McRaeBoisvertExperiment

logger = logging.getLogger(__name__)


class RunExperiment(object):
    def __init__(self, data_path=options['data_path'],
                 mcrae_path=options['mcrae_path'],
                 words_path=options['words_path'],
                 input_dims=options['input_dims'],
                 binomial_probability=options['binomial_probability'],
                 epochs=options['epochs'],
                 ticks=options['nb_ticks']):
        self.data_path = data_path
        self.mcrae_path = os.path.join(self.data_path, mcrae_path)
        self.words_path = os.path.join(self.data_path, words_path)
        self.input_dims = input_dims
        self.binomial_probability = binomial_probability
        self.epochs = epochs
        self.ticks = ticks

    def trainModel(self):
        logger.info('Loading experiment...')
        self.mcrae = McRaeBoisvertExperiment(self.mcrae_path, self.words_path)
        nb_words = len(self.mcrae.include)
        logger.info('Getting random input for %d words, %d dimensions \
with %f probability' % (nb_words, self.input_dims, self.binomial_probability))
        self.X = RandomInput(nb_words, self.input_dims,
                             prob=self.binomial_probability)

        logger.info('Initializing the network...')
        self.attractor = Attractor(self.input_dims,
                                   self.mcrae.experiment.matrix.shape[1],
                                   batch_size=nb_words)
        logger.info('Starting training...')
        self.attractor.fit(self.X.matrix, self.mcrae.experiment.matrix,
                           nb_epochs=self.epochs)

    def priming(self):
        nb_ticks = (2 * self.ticks) - 1
        curves = []
    
        for label in ['Similar', 'Dissimilar']:
            pairs = map(
                lambda (x, y): (self.mcrae.experiment.dicts[0][x],
                                self.mcrae.experiment.dicts[0][y],),
                zip(self.mcrae.df[self.mcrae.df['Description'] ==
                                  label]['Primes'],
                    self.mcrae.df[self.mcrae.df['Description'] ==
                                  label]['Targets']))
            pair_vecs = self.X.matrix[pairs]
            mat = np.empty([len(self.mcrae.df[
                self.mcrae.df['Description'] == label]), nb_ticks],
                dtype='float32')

            for i, pair in enumerate(pair_vecs):
                X = self.attractor.get_priming(pair)[:, 0, :]
                mat[i] = np.delete(X.sum(axis=1), [0])
            curves.append([label, mat.mean(axis=0)])

        return dict(curves)
 
    def plotResults(self):
        res = self.priming()
        plt.plot(res['Similar'], label='Similar', marker='o')
        plt.plot(res['Dissimilar'], label='Dissimilar', marker='o')
        plt.legend(loc='lower right')
        plt.show()


def main():
    experiment = RunExperiment()
    experiment.trainModel()
    experiment.plotResults()


if __name__ == '__main__':
    main()
