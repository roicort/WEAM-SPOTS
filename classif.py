import numpy as np
import constants


es = constants.ExperimentSettings()
fname = constants.csv_filename(
    constants.chosen_prefix,es)
chosen = np.genfromtxt(fname, dtype=int, delimiter=',')
for fold in range(constants.n_folds):
    label = chosen[fold,0]
    idx = chosen[fold,1]
    prefix = constants.classification_name(es)
    # prefix = constants.noised_classification_name(es)
    fname = constants.data_filename(prefix, es, fold)
    classif = np.load(fname)
    print(f'{label}, {classif[idx]}', end='')

    msize = 4
    sigmas = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    for s in sigmas:
        suffix = constants.int_suffix(msize, 'msz')
        suffix += constants.float_suffix(s, 'sgm')
        fname = prefix + suffix
        fname = constants.data_filename(fname, es, fold)
        classif = np.load(fname)
        print(f', {classif[idx]}', end = '')
    print('')

