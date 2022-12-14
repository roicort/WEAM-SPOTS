import numpy as np
import constants


es = constants.ExperimentSettings()
fname = constants.csv_filename(
    constants.chosen_prefix,es)
chosen = np.genfromtxt(fname, dtype=int, delimiter=',')
msize = 4
# sigmas = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
sigma_index = 2
n_depths = 6

for fold in range(constants.n_folds):
    label = chosen[fold,0]
    idx = chosen[fold,1]
    # Uncomment for regular dreaming
    # prefix = constants.classification_name(es)
    # Uncomment for noised dreaming
    prefix = constants.noised_classification_name(es)
    fname = constants.data_filename(prefix, es, fold)
    classif = np.load(fname)
    print(f'{label}, {classif[idx]}', end='')
    # Uncomment for noised dreaming
    prefix = constants.classification_name(es) + constants.noised_suffix
    prefix += constants.int_suffix(msize, 'msz')
    fname = constants.csv_filename(prefix, es, fold)
    classif = np.genfromtxt(fname, delimiter=',')
    start = sigma_index*6
    classif = classif[start:start+n_depths]
    print(classif)

