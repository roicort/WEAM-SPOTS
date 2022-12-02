import numpy as np
import constants

domain_sizes = [32, 64, 128, 256, 512]

def accuracy(labels, predictions):
    n = 0
    for l, p in zip(labels, predictions):
        n += (l == p)
    return n/len(labels)

if __name__ == "__main__":
    dir_prefix = 'runs-'
    es = constants.ExperimentSettings()
    for domain in domain_sizes:
        dirname = f'{dir_prefix}{domain}'
        constants.run_path=dirname
        for fold in range(constants.n_folds):
            prefix = constants.labels_prefix + constants.testing_suffix
            filename = constants.data_filename(prefix, es, fold)
            labels = np.load(filename)
            prefix = constants.classification_name(es)
            filename = constants.data_filename(prefix, es, fold)
            predictions = np.load(filename)
            acc = accuracy(labels, predictions)
            print(f'{domain},{fold},{acc}')
