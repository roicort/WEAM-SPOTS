import numpy as np
import tensorflow as tf
import constants
import dataset

es = constants.ExperimentSettings()
model_prefix = constants.model_name(es)
for fold in range(constants.n_folds):
    # Load de encoder
    filename = constants.classifier_filename(model_prefix, es, fold)
    model = tf.keras.models.load_model(filename)
    model.summary()
    suffix = constants.noised_suffix
    features_filename = \
        constants.data_filename(constants.features_prefix + suffix, es, fold)
    features = np.load(features_filename)
    prefix = constants.noised_classification_name(es)
    labels_filename = constants.data_filename(prefix, es, fold)
    labels = np.argmax(model.predict(features), axis=1)
    np.save(labels_filename, labels)
