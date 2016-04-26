import numpy as np


def priming(vectors, mappings, df, fun, n_ticks=40):
    nb_ticks = (2 * n_ticks) - 1
    curves = []
    
    for label in ['Similar', 'Dissimilar']:
        pairs = map(lambda (x, y): (mappings[x],
                                    mappings[y],),
                    zip(df[df['Description'] == label]['Primes'],
                        df[df['Description'] == label]['Targets']))
        pair_vecs = vectors[pairs]
        mat = np.empty([len(df[df['Description'] == label]), nb_ticks],
                       dtype='float32')

        for i, pair in enumerate(pair_vecs):
            X = fun(pair)[:, 0, :]
            mat[i] = np.delete(X.sum(axis=1), [0])
        curves.append([label, mat.mean(axis=0)])

    return dict(curves)
 
