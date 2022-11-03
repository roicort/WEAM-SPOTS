# Copyright [2020] Luis Alberto Pineda Cortés, Rafael Morales Gamboa.
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

import gzip
import numpy as np
import os
import random
import constants


# This code is an abstraction for the MNIST Fashion dataset,
# which uses the Tensorflow API to get it.
columns = 28
rows = 28

class DataSet:
    _TRAINING_SEGMENT = 0
    _FILLING_SEGMENT = 1
    _TESTING_SEGMENT = 2

    def __init__(self, es, fold):
        self.es = es
        self.fold = fold
        self.dataset, self.size = self.load_dataset(constants.data_path) 

    def load_dataset(self, path):
        data_train, labels_train = self.load_mnist(path, kind='train')
        data_test, labels_test = self.load_mnist(path, kind='t10k')
        data = np.concatenate((data_train, data_test), axis=0)
        labels = np.concatenate((labels_train, labels_test), axis=0)
        pairs = self.shuffle(data, labels)
        return self._data_per_label(pairs)

    def load_mnist(self, path, kind='train'):
        """Load MNIST data from `path`"""
        labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(),
                dtype=np.uint8, offset=8)
        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(),
                dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)
        return images, labels

    def shuffle(self, data, labels):
        pairs = [(data[i], labels[i]) for i in range(len(labels))]
        random.shuffle(pairs)
        return pairs

    def _data_per_label(self, pairs):
        dpl = {}
        for (data, label) in pairs:
            if label in dpl.keys():
                dpl[label].append(data)
            else:
                dpl[label] = [data]
        return dpl, len(pairs)

    def get_training_data(self):
        return self.get_segment(self._TRAINING_SEGMENT, self.fold)

    def get_filling_data(self):
        return self.get_segment(self._FILLING_SEGMENT, self.fold)

    def get_testing_data(self):
        return self.get_segment(self._TESTING_SEGMENT, self.fold)

    def get_segment(self, segment, fold):
        s = self._get_segments_per_label(segment, fold)
        return self._data_and_labels(s)

    def _get_segments_per_label(self, segment, fold):
        dpl = {}
        for label in self.dataset.keys():
            total = len(self.dataset[label])
            training = total*constants.nn_training_percent
            filling = total*constants.am_filling_percent
            testing = total*constants.am_testing_percent
            step = total / constants.n_folds
            i = fold * step
            j = i + training
            k = j + filling
            l = k + testing
            i = int(i)
            j = int(j) % total
            k = int(k) % total
            l = int(l) % total
            n, m = None, None
            if segment == self._TRAINING_SEGMENT:
                n, m = i, j
            elif segment == self._FILLING_SEGMENT:
                n, m = j, k
            elif segment == self._TESTING_SEGMENT:
                n, m = k, l
            dpl[label] = \
                constants.get_data_in_range(self.dataset[label], n, m)
        return dpl

    def _data_and_labels(self, separated: dict):
        pairs = []
        for label in separated.keys():
            for datum in separated[label]:
                pairs.append((datum, label))
        random.shuffle(pairs)
        data = np.array([p[0] for p in pairs], dtype=int)
        labels = np.array([p[1] for p in pairs], dtype=int)        
        return data, labels
