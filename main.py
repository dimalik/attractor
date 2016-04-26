import os
import logging

import matplotlib.pyplot as plt

from config import options
from scripts import Attractor, RandomInput, McRaeBoisvertExperiment, priming

logger = logging.getLogger(__name__)


def getCurves():
    mcrae = McRaeBoisvertExperiment(
        os.path.join(options['data_path'], 'mcrae.pkl'),
        os.path.join(options['data_path'], 'words.pkl'))
    nb_words = len(mcrae.include)

    X = RandomInput(nb_words, options['input_dims'],
                    prob=options['binomial_probability'])

    attractor = Attractor(options['input_dims'],
                          mcrae.experiment.matrix.shape[1],
                          batch_size=nb_words)
    attractor.fit(X.matrix, mcrae.experiment.matrix,
                  nb_epochs=options['epochs'])

    return priming(X.matrix, mcrae.experiment.dicts[0],
                        mcrae.df, attractor.get_priming)


def plotResults(res):
    plt.plot(res['Similar'], label='Similar')
    plt.plot(res['Dissimilar'], label='Dissimilar')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    plotResults(getCurves())
    
