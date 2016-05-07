import collections

import pandas as pd

from nltk.corpus import wordnet as wn


class MySynset(object):
    def __init__(self, synset):
        self.synset = wn.synset(synset)

    def flatten(self, l):
        for el in l:
            if isinstance(el, collections.Iterable) and not isinstance(
                    el, basestring):
                for sub in self.flatten(el):
                    yield sub
            else:
                yield el

    @staticmethod
    def hyp(s):
        return s.hypernyms()

    def get_hypernyms(self):
        hyps = self.synset.tree(self.hyp)
        return list(set([x.name() for x in self.flatten(hyps)]))


ss = wn.all_synsets()
ss_n = [x for x in ss if x.pos() == 'n']

ss_n_dict = {synset.name(): MySynset(synset.name()).get_hypernyms()
             for synset in ss_n}

tmp = [pd.DataFrame(zip([k] * len(v), v),
                    columns=['Concept', 'Features'])
       for k, v in ss_n_dict.iteritems()]
tmp = pd.concat(tmp).reset_index()
tmp.to_csv('wordnet_nouns')
