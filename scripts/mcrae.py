import cPickle as pkl

import numpy as np
import pandas as pd


class Model(object):
    pass


class McRaeModel(Model):
    def __init__(self, dicts=None, matrix=None):
        self.dicts = dicts
        self.matrix = matrix

    def getWordMatrix(self, *args):
        ids = []
        for i, arg in enumerate(args):
            try:
                ids.append((i, arg, self.dicts[0][arg],))
            except KeyError:
                print 'Word %s not found... skipping' % arg
        matrix = self.matrix[[x[2] for x in ids], :]
        dicts = [dict([(x[1], x[0],) for x in ids]), self.dicts[1]]
        return McRaeModel(dicts, matrix)

    def save(self, path):
        with file(path, 'wb') as fout:
            pkl.dump([self.dicts, self.matrix], fout, -1)

    def load(self, path):
        with file(path, 'rb') as fin:
            self.dicts, self.matrix = pkl.load(fin)


def loadMcRae(path):
    df = pd.read_csv(path, delimiter='\t')
    dicts = [{v: i for i, v in enumerate(set(df[value].tolist()))}
             for value in ['Concept', 'Feature'] ]
    
    concept_feature_dict = {
        id_: [dicts[1][x]
              for x in df[df['Concept'] == concept]['Feature'].tolist()]
        for concept, id_ in dicts[0].iteritems()}
    
    matrix = np.zeros([len(dicts[0]), len(dicts[1])])
    
    for i in xrange(len(concept_feature_dict)):
        matrix[i, concept_feature_dict[i]] = 1

    return McRaeModel(dicts, matrix)
