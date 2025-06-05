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

###########################################################################

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Dense, Flatten, \
    Reshape, Conv2DTranspose, BatchNormalization, LayerNormalization, SpatialDropout2D, \
    UpSampling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback, TensorBoard
from joblib import Parallel, delayed
import constants
import dataset


from tqdm import tqdm
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
import matplotlib.pyplot as plt
import os

###########################################################################

# Constants for the training process.

batch_size = 64
epochs = 300
patience = 7
truly_training_percentage = 0.80

def conv_block(entry, layers, filters, dropout, first_block = False):
    conv = None
    for i in range(layers):
        if first_block:
            conv = Conv2D(kernel_size =3, padding ='same', activation='relu', 
                filters = filters, input_shape = (dataset.columns, dataset.rows, 1))(entry)
            first_block = False
        else:
            conv = Conv2D(kernel_size =3, padding ='same', activation='relu', 
                filters = filters)(entry)
        entry = BatchNormalization()(conv)
    pool = MaxPool2D(pool_size = 3, strides =2, padding ='same')(entry)
    drop = SpatialDropout2D(0.4)(pool)
    return drop

def get_encoder():
    dropout = 0.1
    input_data = Input(shape=(dataset.columns, dataset.rows, 1))
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
    output = LayerNormalization(name = 'encoded')(output)
    return input_data, output

def get_decoder():
    input_mem = Input(shape=(constants.domain, ))
    width = dataset.columns // 4
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
    return input_mem, output

def get_classifier():
    input_mem = Input(shape=(constants.domain, ))
    dense = Dense(
        constants.domain, activation='relu',
        input_shape=(constants.domain, ))(input_mem)
    drop = Dropout(0.4)(dense)
    dense = Dense(constants.domain, activation='relu')(drop)
    drop = Dropout(0.4)(dense)
    classification = Dense(constants.n_labels,
        activation='softmax', name='classified')(drop)
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
        accuracy = logs.get('classifier_accuracy')
        val_accuracy = logs.get('val_classifier_accuracy')
        rmse = logs.get('decoder_root_mean_squared_error')
        val_rmse = logs.get('val_decoder_root_mean_squared_error')

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
        training_data, training_labels = dataset.get_training(fold)
        testing_data, testing_labels = dataset.get_testing(fold)
        truly_training = int(len(training_labels)*truly_training_percentage)
        validation_data = training_data[truly_training:]
        validation_labels = training_labels[truly_training:]
        training_data = training_data[:truly_training]
        training_labels = training_labels[:truly_training]

        training_labels = to_categorical(training_labels)
        validation_labels = to_categorical(validation_labels)
        testing_labels = to_categorical(testing_labels)

        rmse = tf.keras.metrics.RootMeanSquaredError()
        input_data = Input(shape=(dataset.columns, dataset.rows, 1))

        input_enc, encoded = get_encoder()
        encoder = Model(input_enc, encoded, name='encoder')
        encoder.compile(optimizer = 'adam')
        encoder.summary()
        input_cla, classified = get_classifier()
        classifier = Model(input_cla, classified, name='classifier')
        classifier.compile(
            loss = 'categorical_crossentropy', optimizer = 'adam',
            metrics = ['accuracy'])
        classifier.summary()
        input_dec, decoded = get_decoder()
        decoder = Model(input_dec, decoded, name='decoder')
        decoder.compile(
            optimizer = 'adam', loss = 'mean_squared_error', metrics = [rmse])
        decoder.summary()
        encoded = encoder(input_data)
        decoded = decoder(encoded)
        classified = classifier(encoded)
        full_classifier = Model(inputs=input_data, outputs=classified, name='full_classifier')
        full_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        autoencoder = Model(inputs = input_data, outputs=decoded, name='autoencoder')
        autoencoder.compile(loss='huber', optimizer='adam', metrics=[rmse])

        model = Model(inputs=input_data, outputs=[classified, decoded])
        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                    optimizer='adam',
                    metrics={'classifier': 'accuracy', 'decoder': rmse})
        model.summary()
        
        log_dir = f"logs/tensorboard/{constants.domain}_{prefix}_fold{fold}"
        TensorboardCallback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        history = model.fit(training_data,
                (training_labels, training_data),
                batch_size=batch_size,
                epochs=epochs,
                validation_data= (validation_data,
                    {'classifier': validation_labels, 'decoder': validation_data}),
                callbacks=[
                    EarlyStopping(),
                    ProgressBar(epochs),
                    ReconstructionsSaver(autoencoder, validation_data, constants.domain),
                    TensorboardCallback
                ],
                verbose=0)
        histories.append(history)
        history = full_classifier.evaluate(testing_data, testing_labels, return_dict=True)
        histories.append(history)
        predicted_labels = np.argmax(full_classifier.predict(testing_data), axis=1)
        confusion_matrix += tf.math.confusion_matrix(np.argmax(testing_labels, axis=1), 
            predicted_labels, num_classes=constants.n_labels)
        history = autoencoder.evaluate(testing_data, testing_data, return_dict=True)
        histories.append(history)
        encoder.save(constants.encoder_filename(prefix, es, fold))
        decoder.save(constants.decoder_filename(prefix, es, fold))
        classifier.save(constants.classifier_filename(prefix, es, fold))
        prediction_prefix = constants.classification_name(es)
        prediction_filename = constants.data_filename(prediction_prefix, es, fold)
        np.save(prediction_filename, predicted_labels)
    confusion_matrix = confusion_matrix.numpy()
    totals = confusion_matrix.sum(axis=1).reshape(-1,1)
    return histories, confusion_matrix/totals


def obtain_features(model_prefix, features_prefix, labels_prefix, data_prefix, es):
    """ Generate features for sound segments, corresponding to phonemes.
    
    Uses the previously trained neural networks for generating the features.
    """
    for fold in range(constants.n_folds):
        # Load de encoder
        filename = constants.encoder_filename(model_prefix, es, fold)
        model = tf.keras.models.load_model(filename)
        model.summary()

        training_data, training_labels = dataset.get_training(fold)
        filling_data, filling_labels = dataset.get_filling(fold)
        testing_data, testing_labels = dataset.get_testing(fold)
        noised_data, noised_labels = dataset.get_testing(fold, noised = True)
        settings = [
            (training_data, training_labels, constants.training_suffix),
            (filling_data, filling_labels, constants.filling_suffix),
            (testing_data, testing_labels, constants.testing_suffix),
            (noised_data, noised_labels, constants.noised_suffix),
        ]
        for s in settings:
            data = s[0]
            labels = s[1]
            suffix = s[2]
            features_filename = \
                constants.data_filename(features_prefix + suffix, es, fold)
            labels_filename = \
                constants.data_filename(labels_prefix + suffix, es, fold)
            features = model.predict(data)
            np.save(features_filename, features)
            np.save(labels_filename, labels)


########################################################################################

class ProgressBar(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress = Progress(
            SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()
        )
        self.task_id = None

    def on_train_begin(self, logs=None):
        self.progress.start()
        self.task_id = self.progress.add_task("Entrenando...", total=self.total_epochs)

    def on_epoch_end(self, epoch, logs=None):
        self.progress.update(self.task_id, advance=1)
        # Extrae todas las métricas relevantes
        acc = logs.get('classifier_accuracy', 0)
        loss = logs.get('classifier_loss', 0)
        dec_loss = logs.get('decoder_loss', 0)
        dec_rmse = logs.get('decoder_root_mean_squared_error', 0)
        total_loss = logs.get('loss', 0)
        val_acc = logs.get('val_classifier_accuracy', 0)
        val_loss = logs.get('val_classifier_loss', 0)
        val_dec_loss = logs.get('val_decoder_loss', 0)
        val_dec_rmse = logs.get('val_decoder_root_mean_squared_error', 0)
        val_total_loss = logs.get('val_loss', 0)

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Epoch", justify="right")
        table.add_column("acc")
        table.add_column("loss")
        table.add_column("dec_loss")
        table.add_column("dec_rmse")
        table.add_column("total_loss")
        table.add_column("val_acc")
        table.add_column("val_loss")
        table.add_column("val_dec_loss")
        table.add_column("val_dec_rmse")
        table.add_column("val_total_loss")

        table.add_row(
            f"{epoch+1}/{self.total_epochs}",
            f"{acc:.4f}",
            f"{loss:.4f}",
            f"{dec_loss:.4f}",
            f"{dec_rmse:.4f}",
            f"{total_loss:.4f}",
            f"{val_acc:.4f}",
            f"{val_loss:.4f}",
            f"{val_dec_loss:.4f}",
            f"{val_dec_rmse:.4f}",
            f"{val_total_loss:.4f}",
        )
        self.progress.console.print(table)

    def on_train_end(self, logs=None):
        self.progress.stop()


class ReconstructionsSaver(Callback):
    def __init__(self, autoencoder, data, domain, every_n_epochs=10, output_dir="decoded", log_dir="logs/tensorboard"):
        super().__init__()
        self.autoencoder = autoencoder
        self.data = data
        self.domain = domain
        self.output_dir = output_dir
        self.every_n_epochs = every_n_epochs
        self.writer = tf.summary.create_file_writer(log_dir)
        self.n = 10
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs == 0:

            originals = self.data[:self.n]
            decoded_imgs = self.autoencoder.predict(originals)

            plt.figure(figsize=(20, 4))
            for i in range(self.n):
                # Imagen original
                ax = plt.subplot(2, self.n, i + 1)
                img_orig = (originals[i].reshape(originals.shape[1], originals.shape[2]) * 255).astype(np.uint8)
                plt.imshow(img_orig, cmap="gray")
                plt.title("Original")
                plt.axis("off")
                # Imagen reconstruida
                ax = plt.subplot(2, self.n, i + 1 + self.n)
                img_recon = (decoded_imgs[i].reshape(originals.shape[1], originals.shape[2]) * 255).astype(np.uint8)
                plt.imshow(img_recon, cmap="gray")
                plt.title("Reconstruida")
                plt.axis("off")
            plt.savefig(os.path.join(self.output_dir, f"{self.domain}_epoch_{epoch+1}.png"))
            plt.close()

            if originals.ndim == 3:
                originals = originals[..., np.newaxis]
            if decoded_imgs.ndim == 3:
                decoded_imgs = decoded_imgs[..., np.newaxis]

            with self.writer.as_default():
                tf.summary.image(f"{self.domain}_originals", originals, step=epoch+1, max_outputs=self.n)
                tf.summary.image(f"{self.domain}_reconstructions", decoded_imgs, step=epoch+1, max_outputs=self.n)
            self.writer.flush()