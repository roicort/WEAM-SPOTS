import numpy as np
import constants
import random

es = constants.ExperimentSettings()
prefix = constants.labels_name(es) + constants.testing_suffix
fname = constants.data_filename(prefix, es, 0)
labels = np.load(fname)

prefix = constants.classification_name(es)
fname = constants.data_filename(prefix, es, 0)
classif = np.load(fname)

chosen = np.zeros((constants.n_labels, 2), dtype=int)
for i in range(constants.n_labels):
    n = 0
    for l, c in zip(labels, classif):
        if (random.randrange(10) == 0) and (l == i) and (l == c):
            chosen[i,0] = i
            chosen[i,1] = n
            break
        n += 1
prefix = constants.chosen_prefix
fname = constants.csv_filename(prefix, es)
np.savetxt(fname, chosen, fmt='%d', delimiter=',')

