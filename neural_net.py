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

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Dense, Flatten, \
    Reshape, Conv2DTranspose, BatchNormalization, LayerNormalization, SpatialDropout2D, \
    UpSampling2D
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from joblib import Parallel, delayed
import constants
import dataset as ds

batch_size = 32
epochs = 300
patience = 7
truly_training_percentage = 0.80

def conv_block(entry, layers, filters, dropout, first_block = False):
    conv = None
    for i in range(layers):
        if first_block:
            conv = Conv2D(kernel_size =3, padding ='same', activation='relu', 
                filters = filters, input_shape = (ds.columns, ds.rows, 1))(entry)
            first_block = False
        else:
            conv = Conv2D(kernel_size =3, padding ='same', activation='relu', 
                filters = filters)(entry)
        entry = BatchNormalization()(conv)
    pool = MaxPool2D(pool_size = 3, strides =2, padding ='same')(entry)
    drop = SpatialDropout2D(0.4)(pool)
    return drop

# The number of layers defined in get_encoder.
encoder_nlayers = 40

def get_encoder():
    dropout = 0.1
    input_data = Input(shape=(ds.columns, ds.rows, 1))
    filters = constants.domain // 16
    output = conv_block(input_data, 2, filters, dropout, first_block=True)
    filters *= 2
    dropout += 0.7
    output = conv_block(output, 2, filters, dropout)
    filters *= 2
    dropout += 0.7
    output = conv_block(output, 3, filters, dropout)
    filters *= 2
    dropout += 0.7
    output = conv_block(output, 3, filters, dropout)
    filters *= 2
    dropout += 0.9
    output = conv_block(output, 3, filters, dropout)
    output = Flatten()(output)
    output = LayerNormalization(name = 'encoder')(output)
    return input_data, output

def get_decoder():
    input_mem = Input(shape=(constants.domain, ))
    width = ds.columns // 4
    filters = constants.domain // 2
    dense = Dense(
        width*width*filters, activation = 'relu',
        input_shape=(constants.domain, ) )(input_mem)
    output = Reshape((width, width, filters))(dense)
    filters *= 2
    dropout = 0.4
    for i in range(2):
        trans = Conv2D(kernel_size=3, strides=1,padding='same', activation='relu',
            filters= filters)(output)
        pool = UpSampling2D(size=2)(trans)
        output = SpatialDropout2D(dropout)(pool)
        dropout /= 2.0
        filters = filters // 2 
        output = BatchNormalization()(output)
    output = Conv2D(filters = 1, kernel_size=3, strides=1,activation='sigmoid', padding='same')(output)
    output_img = Rescaling(255.0, name='autoencoder')(output)
    return input_mem, output_img

# The number of layers defined in get_classifier.
classifier_nlayers = 6

def get_classifier():
    input_mem = Input(shape=(constants.domain, ))
    dense = Dense(
        constants.domain, activation='relu',
        input_shape=(constants.domain, ))(input_mem)
    drop = Dropout(0.4)(dense)
    dense = Dense(constants.domain, activation='relu')(drop)
    drop = Dropout(0.4)(dense)
    classification = Dense(constants.n_labels,
        activation='softmax', name='classifier')(drop)
    return input_mem, classification

class EarlyStopping(Callback):
    """ Stop training when the loss gets lower than val_loss.

        Arguments:
            patience: Number of epochs to wait after condition has been hit.
            After this number of no reversal, training stops.
            It starts working after 10% of epochs have taken place.
    """

    def __init__(self):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.prev_val_loss = float('inf')
        self.prev_val_accuracy = 0.0
        self.prev_val_rmse = float('inf')

        # best_weights to store the weights at which the loss crossing occurs.
        self.best_weights = None
        self.start = min(epochs // 20, 3)
        self.wait = 0

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited since loss crossed val_loss.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        accuracy = logs.get('model_1_accuracy')
        val_accuracy = logs.get('val_model_1_accuracy')
        rmse = logs.get('model_2_root_mean_squared_error')
        val_rmse = logs.get('val_model_2_root_mean_squared_error')

        if epoch < self.start:
            self.best_weights = self.model.get_weights()
        elif (loss < val_loss) or (accuracy > val_accuracy) or (rmse < val_rmse):
            self.wait += 1
        elif (val_accuracy > self.prev_val_accuracy):
            self.wait = 0
            self.prev_val_accuracy = val_accuracy
            self.best_weights = self.model.get_weights()
        elif (val_rmse < self.prev_val_rmse):
            self.wait = 0
            self.prev_val_rmse = val_rmse
            self.best_weights = self.model.get_weights()            
        elif (val_loss < self.prev_val_loss):
            self.wait = 0
            self.prev_val_loss = val_loss
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
        print(f'Epochs waiting: {self.wait}')
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print("Restoring model weights from the end of the best epoch.")
            self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


def train_network(prefix, es):
    confusion_matrix = np.zeros((constants.n_labels, constants.n_labels))
    histories = []
    for fold in range(constants.n_folds):
        dataset = ds.DataSet(es, fold)
        training_data, training_labels = dataset.get_training_data()
        filling_data, filling_labels = dataset.get_filling_data()
        testing_data, testing_labels = dataset.get_testing_data()
        noised_data = ds.noised(testing_data, constants.noise_percent)

        suffixes = {
            constants.training_suffix: (training_data, training_labels),
            constants.filling_suffix : (filling_data, filling_labels),
            constants.testing_suffix : (testing_data, testing_labels),
            constants.noised_suffix  : (noised_data, testing_labels)
        }
            
        for suffix in suffixes:
            data_filename = constants.data_filename(constants.data_prefix+suffix, es, fold)
            labels_filename = constants.data_filename(constants.labels_prefix+suffix, es, fold)
            data, labels = suffixes[suffix]
            np.save(data_filename, data)
            np.save(labels_filename, labels)    

        truly_training = int(len(training_labels)*truly_training_percentage)
        validation_data = training_data[truly_training:]
        validation_labels = training_labels[truly_training:]
        training_data = training_data[:truly_training]
        training_labels = training_labels[:truly_training]

        training_labels = to_categorical(training_labels)
        validation_labels = to_categorical(validation_labels)
        testing_labels = to_categorical(testing_labels)

        rmse = tf.keras.metrics.RootMeanSquaredError()
        input_data = Input(shape=(ds.columns, ds.rows, 1))

        input_enc, encoded = get_encoder()
        encoder = Model(input_enc, encoded)
        encoder.compile(optimizer = 'adam')
        encoder.summary()
        input_cla, classified = get_classifier()
        classifier = Model(input_cla, classified)
        classifier.compile(
            loss = 'categorical_crossentropy', optimizer = 'adam',
            metrics = 'accuracy')
        classifier.summary()
        input_dec, decoded = get_decoder()
        decoder = Model(input_dec, decoded)
        decoder.compile(
            optimizer = 'adam', loss = 'mean_squared_error', metrics = rmse)
        decoder.summary()
        encoded = encoder(input_data)
        decoded = decoder(encoded)
        classified = classifier(encoded)
        full_classifier = Model(inputs=input_data, outputs=classified)
        full_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 'accuracy') 
        autoencoder = Model(inputs = input_data, outputs=decoded)
        autoencoder.compile(loss='huber', optimizer='adam', metrics=rmse)

        model = Model(inputs=input_data, outputs=[classified, decoded])
        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                    optimizer='adam',
                    metrics={'model_1': 'accuracy', 'model_2': rmse})
        model.summary()
        history = model.fit(training_data,
                (training_labels, training_data),
                batch_size=batch_size,
                epochs=epochs,
                validation_data= (validation_data,
                    {'model_1': validation_labels, 'model_2': validation_data}),
                callbacks=[EarlyStopping()],
                verbose=2)
        histories.append(history)
        history = full_classifier.evaluate(testing_data, testing_labels, return_dict=True)
        histories.append(history)
        predicted_labels = full_classifier.predict(testing_data)
        confusion_matrix += tf.math.confusion_matrix(np.argmax(testing_labels, axis=1), 
            np.argmax(predicted_labels, axis=1), num_classes=constants.n_labels)
        history = autoencoder.evaluate(testing_data, testing_data, return_dict=True)
        histories.append(history)
        encoder.save(constants.encoder_filename(prefix, es, fold))
        decoder.save(constants.decoder_filename(prefix, es, fold))
        classifier.save(constants.classifier_filename(prefix, es, fold))
    confusion_matrix = confusion_matrix.numpy()
    totals = confusion_matrix.sum(axis=1).reshape(-1,1)
    return histories, confusion_matrix/totals


def obtain_features(model_prefix, features_prefix, labels_prefix, data_prefix, es):
    """ Generate features for sound segments, corresponding to phonemes.
    
    Uses the previously trained neural networks for generating the features.
    """
    for fold in range(constants.n_folds):
        suffix = constants.training_suffix
        training_features_prefix = features_prefix + suffix        
        training_features_filename = constants.data_filename(training_features_prefix, es, fold)
        training_data_prefix = data_prefix + suffix
        training_data_filename = constants.data_filename(training_data_prefix, es, fold)
        training_data = np.load(training_data_filename)

        suffix = constants.filling_suffix
        filling_features_prefix = features_prefix + suffix        
        filling_features_filename = constants.data_filename(filling_features_prefix, es, fold)
        filling_data_prefix = data_prefix + suffix
        filling_data_filename = constants.data_filename(filling_data_prefix, es, fold)
        filling_data = np.load(filling_data_filename)

        suffix = constants.testing_suffix
        testing_features_prefix = features_prefix + suffix        
        testing_features_filename = constants.data_filename(testing_features_prefix, es, fold)
        testing_data_prefix = data_prefix + suffix
        testing_data_filename = constants.data_filename(testing_data_prefix, es, fold)
        testing_data = np.load(testing_data_filename)

        suffix = constants.noised_suffix
        noised_features_prefix = features_prefix + suffix        
        noised_features_filename = constants.data_filename(noised_features_prefix, es, fold)
        noised_data_prefix = data_prefix + suffix
        noised_data_filename = constants.data_filename(noised_data_prefix, es, fold)
        noised_data = np.load(noised_data_filename)

        # Load de encoder
        filename = constants.encoder_filename(model_prefix, es, fold)
        model = tf.keras.models.load_model(filename)
        model.summary()

        training_features = model.predict(training_data)
        filling_features = model.predict(filling_data)
        testing_features = model.predict(testing_data)
        noised_features = model.predict(noised_data)

        np.save(training_features_filename, training_features)
        np.save(filling_features_filename, filling_features)
        np.save(testing_features_filename, testing_features)
        np.save(noised_features_filename, noised_features)
