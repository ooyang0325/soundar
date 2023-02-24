import numpy as np
import json
import csv
import random
import const_value
import math
from binaural_cues import get_ILD, get_ITD

dataset_path = '../dataset/'

def get_json_data(freq):
    """ load json file

    Args:
        freq (const_value.frequency): frequency

    Returns:
        dict: json data
    """
    filename = dataset_path + f'{freq}_delay/loc.json'
    with open(filename) as json_file:
        data = json.load(json_file)
    return data

def train_test_split(train_data_dict, train_data, train_labels, test_split_ratio=0.2):
    """Split the dataset into train and test set.

    Args:
        train_data (np.array): training data
        train_labels (np.array): training labels
        test_split_ratio (float, optional): split training data for validation. Defaults to 0.2.

    Returns:
        (training_data_dict, training_data, training_label), 
        (validation_data_dict, validation_data, validation_label)
    """

    test_split_count = int(len(train_data) * test_split_ratio)
    new_train_data_dict = train_data_dict[:-test_split_count]
    new_train_data = train_data[:-test_split_count]
    new_train_labels = train_labels[:-test_split_count]
    test_data_dict = train_data_dict[-test_split_count:]    
    test_data = train_data[-test_split_count:]
    test_labels = train_labels[-test_split_count:]
    return (new_train_data_dict, new_train_data, new_train_labels), (test_data_dict, test_data, test_labels)

def get_random_data(freq, data_count=3900, rand=True):
    """ get random data from the dataset

    Args:
        freq (const_value.frequency): frequency
        data_count (int, optional): 3900 is the count of all data. Defaults to 3900.

    Returns:
        list of dict: having data_count of data
    """


    data = get_json_data(freq)

    total_count = len(data['x'])
    full_data_list = []

    for i in range(total_count):
        tmp = {}
        for key in data.keys():
            tmp[key] = data[key][i]
        full_data_list.append(tmp)

    if rand:
        random.shuffle(full_data_list)

    return full_data_list[:data_count]

def get_train_data(freq=const_value.frequency[0], data_count=3900):
    """ get training data and labels

    New features to be added:
        - Choosing the features to be used as training data

    Args:
        freq (const_value.frequency, optional): frequency. Defaults to const_value.frequency[0].
        data_count (int, optional): Defaults to 3900.

    Returns:
        list of dict, np.array, np.array: training data, training labels 
    """
    data = get_random_data(freq, data_count)

    ILD_list = get_ILD(dataset_path + f'{freq}_delay/', data)
    ITD_list = get_ITD(dataset_path + f'{freq}_delay/', data)
    train_data = []
    theta = []

    for i, d in enumerate(data):
        train_data.append([ILD_list[i], ITD_list[i]])
        theta.append(d['theta'])

    train_data = np.array(train_data)
    true_label = np.array(theta)

    return data, train_data, true_label


if __name__ == '__main__':
    data_dict, train_data, true_label = get_train_data(freq=const_value.frequency[0], data_count=50)
    # print(train_data.shape, true_label.shape)
    print(train_data[:5], true_label[:5])

    (x, label), (valid_x, valid_label) = train_test_split(train_data, true_label)
    print(x.shape, label.shape, valid_x.shape, valid_label.shape)


    pass

