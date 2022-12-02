import numpy as np
import constants

domain_sizes = [32, 64, 128, 256, 512]
names = ['memory_entropy', 'memory_precision', 'memory_recall']

def print_row(domain, fname, data):
    print(f'{domain},{fname}', end='')
    for d in data:
        print(f',{d}', end='')
    print('')

if __name__ == "__main__":
    dir_prefix = 'runs-'
    es = constants.ExperimentSettings()
    for domain in domain_sizes:
        dirname = f'{dir_prefix}{domain}'
        constants.run_path=dirname
        for fname in names:
            filename = constants.csv_filename(fname, es)
            data = np.genfromtxt(filename, delimiter=',')
            means = np.mean(data, axis=0)
            stdvs = np.std(data, axis=0)
            print_row(domain, fname, means)
            print_row(domain, fname, stdvs)