import os
import sys

sys.path.insert(0, os.path.dirname(os.getcwd()))

from scripts import Attractor
from data import RandomInput, McRaeBoisvertExperiment

INPUT_DIMS = 30
DATA_PATH = ''


def main():
    mcrae = McRaeBoisvertExperiment(
        os.path.join(DATA_PATH, 'mcrae.pkl'),
        os.path.join(DATA_PATH, 'words.pkl'))

    N_WORDS = len(mcrae.include)
    
    X = RandomInput(N_WORDS, INPUT_DIMS, prob=0.3).matrix

    attractor = Attractor(INPUT_DIMS, mcrae.matrix.shape[1])
    attractor.fit(X, mcrae.matrix, batch_size=N_WORDS)

