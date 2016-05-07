#!/usr/bin/env python
import logging

from config import options
from scripts import Attractor, getRandomInput, ConceptFeatureModel

logger = logging.getLogger(__name__)


def main():
    y = ConceptFeatureModel('data/mcrae/concept_feature_map.tsv',
                            force_dims=2526)
    X = getRandomInput(y.nb_concepts, options['input_dims'],
                       options['binomial_probability'],
                       options['max_iterations'],
                       y.concept_list)
    attractor = Attractor(options['input_dims'], y.matrix.shape[1],
                          batch_size=y.nb_concepts, tau=options['tau'],
                          n_ticks=options['nb_ticks'],
                          train_for=options['train_for'])
    attractor.fit(X.matrix, y.matrix, nb_epochs=2000)

    runExperiment(attractor, X, 'priming')


def runExperiment(model, inputs, type_='association'):
    if type_ == 'association':
        from scripts import AssociationExperiment
        association = AssociationExperiment(
            input_mappings=inputs, nb_ticks=options['nb_ticks'],
            model=model,
            csvfile='data/williams_2005/association_data.tsv')
        association.plotResults()
    elif type_ == 'priming':
        from scripts import PrimingExperiment
        priming = PrimingExperiment(
            input_mappings=inputs, nb_ticks=options['nb_ticks'],
            model=model,
            csvfile='data/mcrae/priming_data.tsv')
        priming.plotResults()


if __name__ == '__main__':
    main()
