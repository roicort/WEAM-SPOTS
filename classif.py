import numpy as np
import constants

indexes = [5494, 6694, 6645, 4874, 6385, 2695, 252, 6165, 198, 257]

es = constants.ExperimentSettings()
# prefix = constants.classification_name(es)
prefix = constants.noised_classification_name(es)
fname = constants.data_filename(prefix, es, 0)
classif = np.load(fname)
print('Classifier: ', end = '')
for i in indexes:
    print(f'\t{classif[i]}', end='')
print('\n')

msize = 4
sigmas = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
for s in sigmas:
    suffix = constants.int_suffix(msize, 'msz')
    suffix += constants.float_suffix(s, 'sgm')
    fname = prefix + suffix
    fname = constants.data_filename(fname, es, 0)
    classif = np.load(fname)
    print(f'Msize: {msize}, Sigma: {s}:', end = '')
    for i in indexes:
        print(f'\t{classif[i]}', end='')
    print('\n')

