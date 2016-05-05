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

    
    def __init__(self, path=None, dicts=None, matrix=None, delim='\t',
                 matrix_dtype='int8'):
        self.path = path
        self.dicts = dicts
        self.matrix = matrix
        self.delim = delim
        self.matrix_dtype = matrix_dtype

        if self.dicts is None and self.matrix is None:
            logging.info('`path` was provided. Loading the csv file...')
            self.dicts, self.matrix = self.loadfromPath(self.delim)


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
        matrix = np.zeros([len(dicts[0]), len(dicts[1])],
                          dtype=self.matrix_dtype)

        logging.info('Populating matrix')
        for i in xrange(len(concept_feature_dict)):
            matrix[i, concept_feature_dict[i]] = 1

        return dicts, matrix

    def getWordMatrix(self, *args):
        ids = []
        for i, arg in enumerate(args):
            try:
                ids.append((i, arg, self.dicts[0][arg],))
            except KeyError:
                logging.warning('Word %s not found... skipping' % arg)
        matrix = self.matrix[[x[2] for x in ids], :].astype(self.matrix_dtype)
        dicts = [dict([(x[1], x[0],) for x in ids]), self.dicts[1]]
        return ConceptFeatureModel(dicts=dicts, matrix=matrix)

    def save(self, path):
        with file(path, 'wb') as fout:
            pkl.dump([self.dicts, self.matrix], fout, -1)

    def load(self, path):
        with file(path, 'rb') as fin:
            self.dicts, self.matrix = pkl.load(fin)
