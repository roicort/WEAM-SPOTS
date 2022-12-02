import json

domain_sizes = [32, 64, 128, 256, 512]
class_metric = 'accuracy'
autor_metric = 'decoder_root_mean_squared_error'

def print_keys(data):
    print('Keys: [ ', end='')
    for k in data.keys():
        print(f'{k}, ', end='')
    print(']')

if __name__ == "__main__":
    prefix = 'runs-'
    suffix = '/model-classifier.json'
    for domain in domain_sizes:
        filename = f'{prefix}{domain}{suffix}'
        # Opening JSON file
        with open(filename, 'r') as f:
            data = json.load(f)
            history = data['history']
            # In every three, the first element is the trace of the training, 
            # and it is ignored. The second and third elements contain
            # the metric and loss for the classifier and autoencoder,
            # respectively
            print(f'History lenght: {len(history)}')
            for i in range(0, len(history), 3):
                class_value = history[i+1][class_metric]
                autor_value = history[i+2][autor_metric]
                print(f'{domain},{class_value},{autor_value}')