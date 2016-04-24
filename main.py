import os
import logging

from config import options
from scripts import Attractor, RandomInput, McRaeBoisvertExperiment, priming

logger = logging.getLogger(__name__)


def main():
    mcrae = McRaeBoisvertExperiment(
        os.path.join(options['data_path'], 'mcrae.pkl'),
        os.path.join(options['data_path'], 'words.pkl'))
    N_WORDS = len(mcrae.include)

    X = RandomInput(N_WORDS, options['input_dims'],
                    prob=options['binomial_probability'])

    attractor = Attractor(options['input_dims'],
                          mcrae.experiment.matrix.shape[1], batch_size=N_WORDS)

    attractor.fit(X.matrix, mcrae.experiment.matrix, nb_epochs=5)
    prime_res = priming(X.matrix, mcrae.experiment.dicts[0], mcrae.df,
                        attractor.get_priming)
    return prime_res

# rewrite that
X = main()
import matplotlib.pyplot as plt

plt.plot(X['Similar'], label='Similar')
plt.plot(X['Dissimilar'], label='Dissimilar')

plt.legend('upper left')
plt.show()
