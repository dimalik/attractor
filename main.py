import os
import logging

import matplotlib.pyplot as plt

from config import options
from scripts import Attractor, RandomInput, McRaeBoisvertExperiment, priming

logger = logging.getLogger(__name__)


def getCurves():
    logger.info('Loading experiment...')
    mcrae = McRaeBoisvertExperiment(
        os.path.join(options['data_path'], 'mcrae.pkl'),
        os.path.join(options['data_path'], 'words.pkl'))
    nb_words = len(mcrae.include)
    logger.info('Getting random input for %d words, %d dimensions \
with %f probability' % (nb_words, options['input_dims'],
                        options['binomial_probability']))
    X = RandomInput(nb_words, options['input_dims'],
                    prob=options['binomial_probability'])
    logger.info('Initializing the network...')
    attractor = Attractor(options['input_dims'],
                          mcrae.experiment.matrix.shape[1],
                          batch_size=nb_words)
    logger.info('Starting training...')
    attractor.fit(X.matrix, mcrae.experiment.matrix,
                  nb_epochs=options['epochs'])

    return priming(X.matrix, mcrae.experiment.dicts[0],
                   mcrae.df, attractor.get_priming, options['nb_ticks'])


def plotResults(res):
    plt.plot(res['Similar'], label='Similar', marker='o')
    plt.plot(res['Dissimilar'], label='Dissimilar', marker='o')
    plt.legend(loc='lower right')
    plt.show()

plotResults(getCurves())
    
if __name__ == '__main__':
    plotResults(getCurves())
    
