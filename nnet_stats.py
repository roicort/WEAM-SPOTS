import numpy as np
import constants

domain_sizes = [32, 64, 128, 256, 512]

def accuracy_fn(labels, predictions):
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
        print(f'Domain size: {domain}')
        acc = []
        for fold in range(constants.n_folds):
            prefix = constants.labels_prefix + constants.testing_suffix
            filename = constants.data_filename(prefix, es, fold)
            labels = np.load(filename)
            prefix = constants.classification_name(es)
            filename = constants.data_filename(prefix, es, fold)
            predictions = np.load(filename)
            accuracy = accuracy_fn(labels, predictions)
            acc.append(accuracy)
            print(f'\tFold: {fold}, {accuracy:.3f}')
        acc = np.array(acc)
        average = np.mean(acc)
        print(f'\t\tAverage: {average:.3f}')
        print('')
