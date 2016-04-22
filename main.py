import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.getcwd()))

from scripts import Attractor, RandomInput, McRaeBoisvertExperiment

logger = logging.getLogger(__name__)

INPUT_DIMS = 30
DATA_PATH = os.path.realpath('data/')


def main():
    mcrae = McRaeBoisvertExperiment(
        os.path.join(DATA_PATH, 'mcrae.pkl'),
        os.path.join(DATA_PATH, 'words.pkl'))

    N_WORDS = len(mcrae.include)

    X = RandomInput(N_WORDS, INPUT_DIMS, prob=0.3)

    attractor = Attractor(INPUT_DIMS, mcrae.experiment.matrix.shape[1])
    attractor.fit(X.matrix, mcrae.experiment.matrix, batch_size=N_WORDS)
