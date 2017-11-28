import pandas as pd
from collections import namedtuple


def import_labeled_csv_data(filename, feature_cols, label_cols):
    """
    Imports .csv file with labeled data and returns preprocessed data tuple.

    Inputs
    ------
    filename:       filename with .csv (str), e.g. 'test.csv'
    feature_cols:   list of indices of columns with features, e.g. [0, 1]
    label_cols:     list of indices of columns with labels, e.g. [2, 3]

    Output
    ------
    data:           named tuple where

    data.features:      features, (np array with (samples, features))
    data.labels:        labels (np array with (samples, labels))
    data.nb_features:   # features (int)
    data.nb_labels:     # labels (int)
    data.nb_samples:    # data samples (int)
    """

    data = namedtuple('data_set', ['features', 'labels', 'nb_features', 
                      'nb_labels', 'nb_samples'])

    data.features = pd.read_csv(filename, skiprows=0, usecols=feature_cols).values

    data.labels = pd.read_csv(filename, skiprows=0, usecols=label_cols).values

    data.nb_samples, data.nb_features = data.features.shape

    data.nb_labels = data.labels.shape[1]

    return data


def create_log_df(filename, layer_size):
    """
    Creates and returns a Pandas DataFrame Object that will be used to log
    training progress across training and validation data.

    Input
    -----
    filename:   filename (str)
    layer_size: number of hidden layers (int)

    Output
    ------
    log_df      Log File (Pandas DataFrame)

    """

    hidden_cols = ['units_hidden%i' %i for i in range(1, layer_size + 1)]
    cols = hidden_cols + ['learning_rate' , 'epoch', 'training_loss', 
                          'training_acc2pc', 'training_acc1pc', 
                          'validation_loss', 'validation_acc2pc', 
                          'validation_acc1pc']

    log_df = pd.DataFrame(columns=cols)

    return log_df




