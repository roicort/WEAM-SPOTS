# Copyright [2020] Luis Alberto Pineda Cortés, Gibrán Fuentes Pineda,
# Rafael Morales Gamboa.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entropic Associative Memory Experiments

Usage:
  eam -h | --help
  eam (-n | -f | -c | -e <experiment> | -o) [--runpath=<runpath>] [ -l (en | es) ]

Options:
  -h        Show this screen.
  -n        Trains the neural network (classifier+autoencoder).
  -f        Generates Features for all data using the encoder.
  -c        Generates graphs Characterizing classes of features (by label).
  -e        Run the experiment (options 1 or 2).
  -o        Generate images from testing data and memories.
  --runpath=<runpath>           Sets the path to the directory where everything will be saved [default: runs]
  -l        Chooses Language for graphs.

The parameter <stage> indicates the stage of learning from which data is used.
"""

import typing
import seaborn
import json
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
from joblib import Parallel, delayed
import numpy as np
import tensorflow as tf
from itertools import islice
import gettext
if typing.TYPE_CHECKING:
    def _(message):
        pass
import gc
from docopt import docopt
import sys
sys.setrecursionlimit(10000)

import neural_net
import dataset
import constants
from associative import AssociativeMemory, AssociativeMemorySystem

# Translation
gettext.install('eam', localedir=None, codeset=None, names=None)


def plot_pre_graph(pre_mean, rec_mean, ent_mean, pre_std, rec_std,
               	   es, tag='', xlabels=constants.memory_sizes,
                   xtitle=None, ytitle=None):

    plt.clf()
    plt.figure(figsize=(6.4, 4.8))

    full_length = 100.0
    step = 0.1
    main_step = full_length/len(xlabels)
    x = np.arange(0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + step

    # Gives space to fully show markers in the top.
    ymax = full_length + 2

    # Replace undefined precision with 1.0.
    pre_mean = np.nan_to_num(pre_mean, copy=False, nan=100.0)

    plt.errorbar(x, pre_mean, fmt='r-o', yerr=pre_std, label=_('Precision'))
    plt.errorbar(x, rec_mean, fmt='b--s', yerr=rec_std, label=_('Recall'))
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.xticks(x, xlabels)

    if xtitle is None:
        xtitle = _('Range Quantization Levels')
    if ytitle is None:
        ytitle = _('Percentage')

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend(loc=4)
    plt.grid(True)

    entropy_labels = [str(e) for e in np.around(ent_mean, decimals=1)]

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'mycolors', ['cyan', 'purple'])
    Z = [[0, 0], [0, 0]]
    levels = np.arange(0.0, xmax, step)
    CS3 = plt.contourf(Z, levels, cmap=cmap)

    cbar = plt.colorbar(CS3, orientation='horizontal')
    cbar.set_ticks(x)
    cbar.ax.set_xticklabels(entropy_labels)
    cbar.set_label(_('Entropy'))

    s = tag + 'graph_prse_MEAN' + _('-english')
    graph_filename = constants.picture_filename(s, es)
    plt.savefig(graph_filename, dpi=600)


def plot_size_graph(response_size, size_stdev, es):
    plt.clf()

    full_length = 100.0
    step = 0.1
    main_step = full_length/len(response_size)
    x = np.arange(0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + step
    ymax = constants.n_labels

    plt.errorbar(x, response_size, fmt='g-D', yerr=size_stdev,
                 label=_('Average number of responses'))
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.xticks(x, constants.memory_sizes)
    plt.yticks(np.arange(0, ymax+1, 1), range(constants.n_labels+1))

    plt.xlabel(_('Range Quantization Levels'))
    plt.ylabel(_('Size'))
    plt.legend(loc=1)
    plt.grid(True)

    graph_filename = constants.picture_filename(
        'graph_size_MEAN' + _('-english'), es)
    plt.savefig(graph_filename, dpi=600)


def plot_behs_graph(no_response, no_correct, correct, es):

    for i in range(len(no_response)):
        total = (no_response[i] + no_correct[i] + correct[i])/100.0
        no_response[i] /= total
        no_correct[i] /= total
        correct[i] /= total

    plt.clf()

    full_length = 100.0
    step = 0.1
    main_step = full_length/len(constants.memory_sizes)
    x = np.arange(0.0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + step
    ymax = full_length
    width = 5       # the width of the bars: can also be len(x) sequence

    plt.bar(x, correct, width, label=_('Correct response'))
    cumm = np.array(correct)
    plt.bar(x, no_correct, width, bottom=cumm, label=_('No correct response'))
    cumm += np.array(no_correct)
    plt.bar(x, no_response, width, bottom=cumm, label=_('No response'))

    plt.xlim(-width, xmax + width)
    plt.ylim(0.0, ymax)
    plt.xticks(x, constants.memory_sizes)

    plt.xlabel(_('Range Quantization Levels'))
    plt.ylabel(_('Labels'))

    plt.legend(loc=0)
    plt.grid(axis='y')

    graph_filename = constants.picture_filename(
        'graph_behaviours_MEAN' + _('-english'), es)
    plt.savefig(graph_filename, dpi=600)


def plot_features_graph(domain, means, stdevs, es):
    """ Draws the characterist shape of features per label.

    The graph is a dots and lines graph with error bars denoting standard deviations.
    """
    ymin = np.PINF
    ymax = np.NINF
    for i in constants.all_labels:
        yn = (means[i] - stdevs[i]).min()
        yx = (means[i] + stdevs[i]).max()
        ymin = ymin if ymin < yn else yn
        ymax = ymax if ymax > yx else yx
    main_step = 100.0 / domain
    xrange = np.arange(0, 100, main_step)
    fmts = constants.label_formats
    for i in constants.all_labels:
        plt.clf()
        plt.figure(figsize=(12, 5))
        plt.errorbar(xrange, means[i], fmt=fmts[i],
                     yerr=stdevs[i], label=str(i))
        plt.xlim(0, 100)
        plt.ylim(ymin, ymax)
        plt.xticks(xrange, labels='')
        plt.xlabel(_('Features'))
        plt.ylabel(_('Values'))
        plt.legend(loc='right')
        plt.grid(True)
        filename = constants.features_name(
            es) + '-' + str(i).zfill(3) + _('-english')
        plt.savefig(constants.picture_filename(filename, es), dpi=600)


def plot_conf_matrix(matrix, tags, prefix, es):
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))
    seaborn.heatmap(matrix, xticklabels=tags, yticklabels=tags,
                    vmin=0.0, vmax=1.0, annot=False, cmap='Blues')
    plt.xlabel(_('Prediction'))
    plt.ylabel(_('Label'))
    filename = constants.picture_filename(prefix, es)
    plt.savefig(filename, dpi=600)


def plot_memory(memory: AssociativeMemory, prefix, es, fold):
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))
    seaborn.heatmap(memory.relation/memory.max_value, vmin=0.0, vmax=1.0,
                    annot=False, cmap='coolwarm')
    plt.xlabel(_('Characteristics'))
    plt.ylabel(_('Values'))
    filename = constants.picture_filename(prefix, es, fold)
    plt.savefig(filename, dpi=600)


def plot_memories(ams, es, fold):
    for label in ams:
        prefix = f'memory-{label}-state'
        plot_memory(ams[label], prefix, es, fold)


def get_label(memories, weights=None, entropies=None):
    if len(memories) == 1:
        return memories[0]
    random.shuffle(memories)
    if (entropies is None) or (weights is None):
        return memories[0]
    else:
        i = memories[0]
        entropy = entropies[i]
        weight = weights[i]
        penalty = entropy/weight if weight > 0 else float('inf')
        for j in memories[1:]:
            entropy = entropies[j]
            weight = weights[j]
            new_penalty = entropy/weight if weight > 0 else float('inf')
            if new_penalty < penalty:
                i = j
                penalty = new_penalty
        return i


def msize_features(features, msize, min_value, max_value):
    return np.round((msize-1)*(features-min_value) / (max_value-min_value)).astype(int)

def rsize_recall(recall, msize, min_value, max_value):
    if (msize == 1):
        return (recall.astype(dtype=float) + 1.0)*(max_value - min_value)/2
    else:
        return (max_value - min_value)* recall.astype(dtype=float) \
            /(msize - 1.0) + min_value


TP = (0, 0)
FP = (0, 1)
FN = (1, 0)
TN = (1, 1)


def conf_sum(cms, t):
    return np.sum([cms[i][t] for i in range(len(cms))])


def memories_precision(cms):
    total = conf_sum(cms, TP) + conf_sum(cms, FN)
    if total == 0:
        return 0.0
    precision = 0.0
    for m in range(len(cms)):
        denominator = (cms[m][TP] + cms[m][FP])
        if denominator == 0:
            m_precision = 1.0
        else:
            m_precision = cms[m][TP] / denominator
        weight = (cms[m][TP] + cms[m][FN]) / total
        precision += weight*m_precision
    return precision


def memories_recall(cms):
    total = conf_sum(cms, TP) + conf_sum(cms, FN)
    if total == 0:
        return 0.0
    recall = 0.0
    for m in range(len(cms)):
        m_recall = cms[m][TP] / (cms[m][TP] + cms[m][FN])
        weight = (cms[m][TP] + cms[m][FN]) / total
        recall += weight*m_recall
    return recall


def memories_accuracy(cms):
    total = conf_sum(cms, TP) + conf_sum(cms, FN)
    if total == 0:
        return 0.0
    accuracy = 0.0
    for m in range(len(cms)):
        m_accuracy = (cms[m][TP] + cms[m][TN]) / total
        weight = (cms[m][TP] + cms[m][FN]) / total
        accuracy += weight*m_accuracy
    return accuracy


def register_in_memory(memory, features_iterator):
    for features in features_iterator:
        memory.register(features)


def memory_entropy(m, memory: AssociativeMemory):
    return m, memory.entropy


def recognize_by_memory(eam, tef_rounded, tel, msize, minimum, maximum, classifier):
    data = []
    labels = []
    confrix = np.zeros(
        (constants.n_labels, constants.n_labels), dtype='int')
    behaviour = np.zeros(constants.n_behaviours, dtype=np.float64)
    unknown = 0
    for features, label in zip(tef_rounded, tel):
        memory, recognized, _ = eam.recall(features)
        if recognized:
            mem = rsize_recall(memory, msize, minimum, maximum)
            data.append(mem)
            labels.append(label)
        else:
            unknown += 1
    data = np.array(data)
    predictions = np.argmax(classifier.predict(data), axis=1)
    for correct, prediction in zip(labels, predictions):
        # For calculation of per memory precision and recall
        confrix[correct, prediction] += 1
    behaviour[constants.no_response_idx] = unknown
    behaviour[constants.correct_response_idx] = \
        np.sum([confrix[i,i] for i in range(constants.n_labels)])

    behaviour[constants.no_correct_response_idx] = \
        len(tel) - unknown - behaviour[constants.correct_response_idx]
    print(f'Confusion matrix:\n{confrix}')
    print(f'Behaviour: {behaviour}')
    return confrix, behaviour


def split_by_label(fl_pairs):
    label_dict = {}
    for label in range(constants.n_labels):
        label_dict[label] = []
    for features, label in fl_pairs:
        label_dict[label].append(features)
    return label_dict.items()


def split_every(n, iterable):
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))


def _get_f1(t):
  return t[0]

def optimum_indexes(precisions, recalls):
    f1s = []
    i = 0
    for p, r in zip(precisions, recalls):
        f1 = 0 if (r+p) == 0 else 2*(r*p)/(r+p)
        f1s.append((f1, i))
        i += 1
    f1s.sort(reverse = True, key = _get_f1)
    return [t[1] for t in f1s[:constants.n_best_memory_sizes]]

def get_ams_results(
        midx, msize, domain, trf, tef, trl, tel, classifier, es, fold):
    # Round the values
    max_value = trf.max()
    other_value = tef.max()
    max_value = max_value if max_value > other_value else other_value
    min_value = trf.min()
    other_value = tef.min()
    min_value = min_value if min_value < other_value else other_value

    trf_rounded = msize_features(trf, msize, min_value, max_value)
    tef_rounded = msize_features(tef, msize, min_value, max_value)
    behaviour = np.zeros(constants.n_behaviours, dtype=np.float64)

    # Create the memory.
    p = es.mem_params
    eam = AssociativeMemory(
        domain, msize, p[constants.xi_idx], p[constants.sigma_idx],
        p[constants.iota_idx], p[constants.kappa_idx])

    # Registrate filling data.
    for features in trf_rounded:
        eam.register(features)

    # Recognize test data.
    confrix, behaviour = recognize_by_memory(
        eam, tef_rounded, tel, msize, min_value, max_value, classifier)
    responses = len(tel) - behaviour[constants.no_response_idx]
    precision = behaviour[constants.correct_response_idx]/float(responses)
    recall = behaviour[constants.correct_response_idx]/float(len(tel))
    behaviour[constants.precision_idx] = precision
    behaviour[constants.recall_idx] = recall
    return midx, eam.entropy, behaviour, confrix


def test_memory_sizes(domain, es):
    all_entropies = []
    precision = []
    recall = []
    all_confrixes = []

    no_response = []
    no_correct_response = []
    correct_response = []

    print('Testing the memory')

    # Retrieve de classifier
    model_prefix = constants.model_name(es)

    for fold in range(constants.n_folds):
        gc.collect()
        filename = constants.classifier_filename(model_prefix, es, fold)
        classifier = tf.keras.models.load_model(filename)
        print(f'Fold: {fold}')
        suffix = constants.filling_suffix
        filling_features_filename = constants.features_name(es) + suffix
        filling_features_filename = constants.data_filename(
            filling_features_filename, es, fold)
        filling_labels_filename = constants.labels_name(es) + suffix
        filling_labels_filename = constants.data_filename(
            filling_labels_filename, es, fold)

        suffix = constants.testing_suffix
        testing_features_filename = constants.features_name(es) + suffix
        testing_features_filename = constants.data_filename(
            testing_features_filename, es, fold)
        testing_labels_filename = constants.labels_name(es) + suffix
        testing_labels_filename = constants.data_filename(
            testing_labels_filename, es, fold)

        filling_features = np.load(filling_features_filename)
        filling_labels = np.load(filling_labels_filename)
        testing_features = np.load(testing_features_filename)
        testing_labels = np.load(testing_labels_filename)

        behaviours = np.zeros(
            (len(constants.memory_sizes), constants.n_behaviours))
        measures = []
        confrixes = []
        entropies = []
        for midx, msize in enumerate(constants.memory_sizes):
            print(f'Memory size: {msize}')
            results = get_ams_results(midx, msize, domain,
                filling_features, testing_features,
                filling_labels, testing_labels, classifier, es, fold)
            measures.append(results)
        for midx, entropy, behaviour, confrix in measures:
            entropies.append(entropy)
            behaviours[midx, :] = behaviour
            confrixes.append(confrix)

        ###################################################################3##
        # Measures by memory size

        # Average entropy among al digits.
        all_entropies.append(entropies)

        # Average precision and recall as percentage
        precision.append(behaviours[:, constants.precision_idx]*100)
        recall.append(behaviours[:, constants.recall_idx]*100)

        all_confrixes.append(np.array(confrixes))
        no_response.append(behaviours[:, constants.no_response_idx])
        no_correct_response.append(
            behaviours[:, constants.no_correct_response_idx])
        correct_response.append(behaviours[:, constants.correct_response_idx])

    # Every row is training fold, and every column is a memory size.
    all_entropies = np.array(all_entropies)
    precision = np.array(precision)
    recall = np.array(recall)
    all_confrixes = np.array(all_confrixes)

    average_entropy = np.mean(all_entropies, axis=0)
    average_precision = np.mean(precision, axis=0)
    stdev_precision = np.std(precision, axis=0)
    average_recall = np.mean(recall, axis=0)
    stdev_recall = np.std(recall, axis=0)
    average_confrixes = np.mean(all_confrixes, axis=0)

    no_response = np.array(no_response)
    no_correct_response = np.array(no_correct_response)
    correct_response = np.array(correct_response)
    main_no_response = np.mean(no_response, axis=0)
    main_no_correct_response = np.mean(no_correct_response, axis=0)
    main_correct_response = np.mean(correct_response, axis=0)
    best_memory_idx = optimum_indexes(average_precision, average_recall)
    best_memory_sizes = [constants.memory_sizes[i] for i in best_memory_idx]
    main_behaviours = \
        [main_no_response, main_no_correct_response, main_correct_response]

    np.savetxt(constants.csv_filename(
        'memory_precision', es), precision, delimiter=',')
    np.savetxt(constants.csv_filename(
        'memory_recall', es), recall, delimiter=',')
    np.savetxt(constants.csv_filename(
        'memory_entropy', es), all_entropies, delimiter=',')
    np.savetxt(constants.csv_filename('main_behaviours', es),
               main_behaviours, delimiter=',')
    np.save(constants.data_filename('memory_confrixes', es), average_confrixes)
    np.save(constants.data_filename('behaviours', es), behaviours)
    plot_pre_graph(average_precision, average_recall, average_entropy,
                   stdev_precision, stdev_recall, es)
    plot_behs_graph(main_no_response, main_no_correct_response, main_correct_response, es)
    print('Memory size evaluation completed!')
    return best_memory_sizes


def test_filling_percent(
        eam, msize, min_value, max_value,
        trf, tef, tel, percent, classifier):
    # Registrate filling data.
    for features in trf:
        eam.register(features)
    print(f'Filling of memories done at {percent}%')
    _, behaviour = recognize_by_memory(
        eam, tef, tel, msize, min_value, max_value, classifier)
    responses = len(tel) - behaviour[constants.no_response_idx]
    precision = behaviour[constants.correct_response_idx]/float(responses)
    recall = behaviour[constants.correct_response_idx]/float(len(tel))
    behaviour[constants.precision_idx] = precision
    behaviour[constants.recall_idx] = recall
    return behaviour, eam.entropy

def test_filling_per_fold(mem_size, domain, es, fold):
    # Create the required associative memories.
    p = es.mem_params
    eam = AssociativeMemory(
        domain, mem_size, p[constants.xi_idx],
        p[constants.sigma_idx], p[constants.iota_idx], p[constants.kappa_idx])
    model_prefix = constants.model_name(es)
    filename = constants.classifier_filename(model_prefix, es, fold)
    classifier = tf.keras.models.load_model(filename)

    suffix = constants.filling_suffix
    filling_features_filename = constants.features_name(es) + suffix
    filling_features_filename = constants.data_filename(
        filling_features_filename, es, fold)
    filling_labels_filename = constants.labels_name(es) + suffix
    filling_labels_filename = constants.data_filename(
        filling_labels_filename, es, fold)

    suffix = constants.testing_suffix
    testing_features_filename = constants.features_name(es) + suffix
    testing_features_filename = constants.data_filename(
        testing_features_filename, es, fold)
    testing_labels_filename = constants.labels_name(es) + suffix
    testing_labels_filename = constants.data_filename(
        testing_labels_filename, es, fold)

    filling_features = np.load(filling_features_filename)
    filling_labels = np.load(filling_labels_filename)
    testing_features = np.load(testing_features_filename)
    testing_labels = np.load(testing_labels_filename)

    filling_min = filling_features.min()
    testing_min = testing_features.min()
    filling_max = filling_features.max()
    testing_max = testing_features.max()
    minimum = filling_min if filling_min < testing_min else testing_min
    maximum = filling_max if filling_max > testing_max else testing_max
    filling_rounded = msize_features(
        filling_features, mem_size, minimum, maximum)
    testing_rounded = msize_features(
        testing_features, mem_size, minimum, maximum)

    total = len(filling_labels)
    percents = np.array(constants.memory_fills)
    steps = np.round(total*percents/100.0).astype(int)

    fold_entropies = []
    fold_precision = []
    fold_recall = []

    start = 0
    for percent, end in zip(percents, steps):
        features = filling_rounded[start:end]
        print(f'Filling from {start} to {end}.')
        behaviour, entropy = \
            test_filling_percent(
                eam, mem_size, minimum, maximum, features,
                testing_rounded, testing_labels, percent, classifier)

        # A list of tuples (position, label, features)
        # fold_recalls += recalls
        # An array with average entropy per step.
        fold_entropies.append(entropy)
        # Arrays with precision, and recall.
        fold_precision.append(behaviour[constants.precision_idx])
        fold_recall.append(behaviour[constants.recall_idx])
        start = end
    # Use this to plot current state of memories
    # as heatmaps.
    # plot_memories(ams, es, fold)
    fold_entropies = np.array(fold_entropies)
    fold_precision = np.array(fold_precision)
    fold_recall = np.array(fold_recall)
    print(f'Filling test completed for fold {fold}')
    return fold, fold_entropies, fold_precision, fold_recall


def test_memory_fills(domain, mem_sizes, es):
    memory_fills = constants.memory_fills
    testing_folds = constants.n_folds
    best_filling_percents = []
    for mem_size in mem_sizes:
        # All entropies, precision, and recall, per size, fold, and fill.
        total_entropies = np.zeros((testing_folds, len(memory_fills)))
        total_precisions = np.zeros((testing_folds, len(memory_fills)))
        total_recalls = np.zeros((testing_folds, len(memory_fills)))
        list_results = []

        for fold in range(testing_folds):
            results = test_filling_per_fold(mem_size, domain, es, fold)
            list_results.append(results)
        for fold, entropies, precisions, recalls in list_results:
            total_precisions[fold] = precisions
            total_recalls[fold] = recalls
            total_entropies[fold] = entropies

        main_avrge_entropies = np.mean(total_entropies, axis=0)
        main_stdev_entropies = np.std(total_entropies, axis=0)
        main_avrge_precisions = np.mean(total_precisions, axis=0)
        main_stdev_precisions = np.std(total_precisions, axis=0)
        main_avrge_recalls = np.mean(total_recalls, axis=0)
        main_stdev_recalls = np.std(total_recalls, axis=0)

        np.savetxt(
            constants.csv_filename(
                'main_average_precision' + constants.numeric_suffix('sze', mem_size), es),
            main_avrge_precisions, delimiter=',')
        np.savetxt(
            constants.csv_filename(
                'main_average_recall' + constants.numeric_suffix('sze', mem_size), es),
            main_avrge_recalls, delimiter=',')
        np.savetxt(
            constants.csv_filename(
                'main_average_entropy' + constants.numeric_suffix('sze', mem_size), es),
            main_avrge_entropies, delimiter=',')
        np.savetxt(
            constants.csv_filename(
                'main_stdev_precision' + constants.numeric_suffix('sze', mem_size), es),
            main_stdev_precisions, delimiter=',')
        np.savetxt(
            constants.csv_filename(
                'main_stdev_recall' + constants.numeric_suffix('sze', mem_size), es),
            main_stdev_recalls, delimiter=',')
        np.savetxt(
            constants.csv_filename(
                'main_stdev_entropy' + constants.numeric_suffix('sze', mem_size), es),
            main_stdev_entropies, delimiter=',')

        plot_pre_graph(main_avrge_precisions*100, main_avrge_recalls*100, main_avrge_entropies,
                    main_stdev_precisions*100, main_stdev_recalls *
                    100, es, 'recall' + constants.numeric_suffix('sze', mem_size),
                    xlabels=constants.memory_fills, xtitle=_('Percentage of memory corpus'))

        bf_idx = optimum_indexes(
            main_avrge_precisions, main_avrge_recalls)
        best_filling_percents.append(constants.memory_fills[bf_idx[0]])
        print(f'Testing fillings for memory size {mem_size} done.')
    return best_filling_percents


def get_all_data(prefix, es):
    data = None
    for fold in range(constants.n_folds):
        filename = constants.data_filename(prefix, es, fold)
        if data is None:
            data = np.load(filename)
        else:
            newdata = np.load(filename)
            data = np.concatenate((data, newdata), axis=0)
    return data


def save_history(history, prefix, es):
    """ Saves the stats of neural networks.

    Neural networks stats may come either as a History object, that includes
    a History.history dictionary with stats, or directly as a dictionary.
    """
    stats = {}
    stats['history'] = []
    for h in history:
        while not ((type(h) is dict) or (type(h) is list)):
            h = h.history
        stats['history'].append(h)
    with open(constants.json_filename(prefix, es), 'w') as outfile:
        json.dump(stats, outfile)


def save_conf_matrix(matrix, prefix, es):
    name = prefix + constants.matrix_suffix
    plot_conf_matrix(matrix, range(constants.n_labels), name, es)
    filename = constants.data_filename(name, es)
    np.save(filename, matrix)


def save_learned_params(mem_sizes, fill_percents, es):
    name = constants.learn_params_name(es)
    filename = constants.data_filename(name, es)
    np.save(filename, np.array([mem_sizes, fill_percents], dtype=int))


##############################################################################
# Main section

def create_and_train_network(es):
    model_prefix = constants.model_name(es)
    stats_prefix = model_prefix + constants.classifier_suffix
    history, conf_matrix = neural_net.train_network(model_prefix, es)
    save_history(history, stats_prefix, es)
    save_conf_matrix(conf_matrix, stats_prefix, es)


def produce_features_from_data(es):
    model_prefix = constants.model_name(es)
    features_prefix = constants.features_name(es)
    labels_prefix = constants.labels_name(es)
    data_prefix = constants.data_name(es)
    neural_net.obtain_features(
        model_prefix, features_prefix, labels_prefix, data_prefix, es)


def create_and_train_autoencoders(es):
    model_prefix = constants.model_name(es)
    stats_prefix = model_prefix + constants.decoder_suffix
    features_prefix = constants.features_name(es)
    data_prefix = constants.data_name(es)
    history = neural_net.train_decoder(
        model_prefix, features_prefix, data_prefix, es)
    save_history(history, stats_prefix, es)


def characterize_features(es):
    """ Produces a graph of features averages and standard deviations.
    """
    features_prefix = constants.features_name(es)
    tf_filename = features_prefix + constants.testing_suffix
    labels_prefix = constants.labels_name(es)
    tl_filename = labels_prefix + constants.testing_suffix
    features = get_all_data(tf_filename, es)
    labels = get_all_data(tl_filename, es)
    d = {}
    for i in constants.all_labels:
        d[i] = []
    for (i, feats) in zip(labels, features):
        # Separates features per label.
        d[i].append(feats)
    means = {}
    stdevs = {}
    for i in constants.all_labels:
        # The list of features becomes a matrix
        d[i] = np.array(d[i])
        means[i] = np.mean(d[i], axis=0)
        stdevs[i] = np.std(d[i], axis=0)
    plot_features_graph(constants.domain, means, stdevs, es)


def run_evaluation(es):
    best_memory_sizes = test_memory_sizes(constants.domain, es)
    print(f'Best memory sizes: {best_memory_sizes}')
    best_filling_percents = test_memory_fills(
        constants.domain, best_memory_sizes, es)
    save_learned_params(best_memory_sizes, best_filling_percents, es)


def generate_output(es):
    neural_net.decode(constants.model_name(es),
                      constants.data_prefix, constants.labels_prefix, constants.features_prefix, es)


if __name__ == "__main__":
    args = docopt(__doc__)

    # Processing language.
    lang = 'en'
    if args['es']:
        lang = 'es'
        es = gettext.translation('eam', localedir='locale', languages=['es'])
        es.install()

    # Processing runpath.
    constants.run_path = args['--runpath']

    prefix = constants.memory_parameters_prefix
    filename = constants.csv_filename(prefix)
    parameters = \
        np.genfromtxt(filename, dtype=float, delimiter=',', skip_header=1)
    exp_settings = constants.ExperimentSettings(parameters)
    print(f'Working directory: {constants.run_path}')
    print(f'Experimental settings: {exp_settings}')

    # PROCESSING OF MAIN OPTIONS.

    if args['-n']:
        create_and_train_network(exp_settings)
    elif args['-f']:
        produce_features_from_data(exp_settings)
    elif args['-c']:
        characterize_features(exp_settings)
    elif args['-e']:
        experiment = int(args['<experiment>'])
        if experiment == 1:
            run_evaluation(exp_settings)
        else:
            print(f'Experiment number not valid: {experiment}')
    elif args['-o']:
        generate_output(exp_settings)
