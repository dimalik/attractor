import numpy as np

def priming(vectors, mappings, df, fun, n_ticks=19):

    curves = []
    
    for label in ['Similar', 'Dissimilar']:
        pairs = map(lambda (x, y): (mappings[x],
                                    mappings[y],),
                    zip(df[df['Description'] == label]['Primes'],
                        df[df['Description'] == label]['Targets']))
        pair_vecs = vectors[pairs]
        
        mat = np.empty([len(df[df['Description'] == label]), n_ticks*2],
                           dtype='float32')

        for i, pair in enumerate(pair_vecs):
            X = fun(pair)[:, 0, :]
            mat[i] = X.sum(axis=1)[([False] + [True] *
                                    (n_ticks)) * 2]
        curve = mat.mean(axis=0)
        curves.append([label, curve])

    return dict(curves)
