import cPickle as pkl
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ConceptFeatureModel(object):
    """Turns a concept-features list to a binary feature matrix.

    Attributes:
        path (str): Path to csv file which has at least two columns
                    one for concepts and one for features.
        dicts (list): If provided then `path` is redundant. `dicts` is
                      a list of two dictionaries (concept->id and feature->id)
        matrix (ndarray): A binary matrix
    """

    def __init__(self, path=None, concepts=None, features=None, matrix=None,
                 delim='\t', force_dims=None, matrix_dtype='float32'):
        self.path = path
        self.concepts = concepts
        self.features = features
        self.matrix = matrix
        self.delim = delim
        self.force_dims = force_dims
        self.matrix_dtype = matrix_dtype

        if self.path is not None:
            logging.info('`path` was provided. Loading the csv file...')
            (self.concepts, self.features,
             self.matrix) = self.loadfromPath(self.delim)

        self.concept_list = self.concepts.keys()
        self.nb_concepts = len(self.concept_list)

    def loadfromPath(self, delimiter, c_header='Concept',
                     f_header='Feature'):
        try:
            df = pd.read_csv(self.path, delimiter=delimiter)
        except IOError:
            logging.error('The path provided was wrong')
        try:
            dicts = [{v: i for i, v in enumerate(set(df[value].tolist()))}
                     for value in [c_header, f_header]]
        except KeyError:
            logging.error('Columns did not exist')

        concept_feature_dict = {
            id_: [dicts[1][x]
                  for x in df[df[c_header] == c][f_header].tolist()]
            for c, id_ in dicts[0].iteritems()}

        logging.info('Initializing matrix')

        dims = self.force_dims if self.force_dims else len(dicts[1])
        perm = np.random.permutation(dims)
        matrix = np.zeros([len(dicts[0]), dims],
                          dtype=self.matrix_dtype)
        matrix = matrix[:, perm]
        logging.info('Populating matrix')
        for i in xrange(len(concept_feature_dict)):
            matrix[i, concept_feature_dict[i]] = 1

        return dicts[0], dicts[1], matrix

    def getWordMatrix(self, *args):
        ids = []
        for i, arg in enumerate(args):
            try:
                ids.append((i, arg, self.concepts[arg],))
            except KeyError:
                logging.warning('Word %s not found... skipping' % arg)
        matrix = self.matrix[[x[2] for x in ids], :].astype(self.matrix_dtype)
        dicts = [dict([(x[1], x[0],) for x in ids]), self.features]
        return ConceptFeatureModel(concepts=dicts[0],
                                   features=dicts[1],
                                   matrix=matrix)

    def save(self, path):
        with file(path, 'wb') as fout:
            pkl.dump([self.concepts, self.features, self.matrix], fout, -1)

    def load(self, path):
        with file(path, 'rb') as fin:
            self.concepts, self.features, self.matrix = pkl.load(fin)
