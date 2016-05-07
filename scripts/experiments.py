import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Experiment(object):
    def __init__(self, model, input_mappings, csvfile):
        self.input_mappings = input_mappings
        self.model = model
        self.df = pd.read_csv(csvfile, delimiter='\t')

    def plotResults(self, *args, **kwargs):
        raise NotImplementedError


class AttractorExperiment(Experiment):
    def __init__(self, nb_ticks, *args, **kwargs):
        self.nb_ticks = nb_ticks
        super(AttractorExperiment, self).__init__(*args, **kwargs)
        self.curves = self.getCurves()

    def getCurves(self, *args, **kwargs):
        raise NotImplementedError

    def plotResults(self):
        labels = self.curves.keys()
        for label in labels:
            plt.plot(self.curves[label], label=label, marker='o')
        plt.legend(loc='lower right')
        plt.show()


class PrimingExperiment(AttractorExperiment):

    def getCurves(self):
        nb_ticks = (2 * self.nb_ticks) - 1
        curves = []

        for label in ['Similar', 'Dissimilar']:
            primes = list(self.df[self.df['Description'] == label]['Prime'])
            targets = list(self.df[self.df['Description'] == label]['Target'])
            noun_pairs = zip(primes, targets)
            pairs = np.array([(self.input_mappings[x], self.input_mappings[y],)
                              for x, y in noun_pairs])
            mat = np.empty([len(primes), nb_ticks], dtype='float32')
            for i, pair in enumerate(pairs):
                X = self.model.get_priming(pair)[:, 0, :]
                mat[i] = np.delete(X.sum(axis=1), [0])
            curves.append([label, mat.mean(axis=0)])
        return dict(curves)


class AssociationExperiment(AttractorExperiment):

    def getCurves(self):
        nb_ticks = (2 * self.nb_ticks) - 1
        curves = []

        for label in ['animate', 'inanimate']:
            nouns = list(self.df[self.df['Description'] == label]['Noun'])
            inputs = [self.input_mappings[x] for x in nouns]
            mat = np.empty([len(nouns), nb_ticks], dtype='float32')
            for i, word in enumerate(inputs):
                X = self.model.get_activation(word)
                mat[i] = np.delete(X.sum(axis=1), [0])
            curves.append([label, mat.mean(axis=0)])
        return dict(curves)
