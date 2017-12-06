import pandas as pd
from collections import namedtuple


def load_labeled_csv(filename, feature_cols, label_cols):
    """
    Load a .csv file with labeled data into memory. Processes data and returns
    a named tuple with relevant data information.

    Arguments:
    ----------
        filename: string.
            Filename with .csv (str), e.g. 'test.csv'.
        feature_cols: list.
            Indices of columns with features, e.g. [0, 1].
        label_cols: list.
            Indices of columns with labels, e.g. [2, 3].

    Returns:
    --------
        data:   named tuple.
            data.features: array-like, shape=[# samples, # features].
                Features of the data set.
            data.labels: array-like, shape=[# samples, # labels].
                Labels of the data set.
            data.nb_features:   integer. 
                Number of features.
            data.nb_labels: integer.
                Number of labels.
            data.nb_samples: integer.
                Number of samples.
    """

    data = namedtuple('data_set', ['features', 'labels', 'nb_features',
                      'nb_labels', 'nb_samples'])

    raw_data = pd.read_csv(filename, skiprows=0).values
    
    data.features = raw_data[:, feature_cols]
    data.labels = raw_data[:, label_cols]

    data.nb_features = len(feature_cols)
    data.nb_labels = len(label_cols)
    data.nb_samples = raw_data.shape[0]

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
    cols = hidden_cols + ['learning_rate' , 'training_loss', 
                          'training_acc2pc', 'training_acc1pc', 
                          'validation_loss', 'validation_acc2pc', 
                          'validation_acc1pc']

    log_df = pd.DataFrame(columns=cols)

    return log_df
    
