import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import linear_model
from sklearn import svm

import tensorflow as tf
import keras
from keras.models import Sequential
from keras import layers

import numpy as np
import math
from get_data import get_train_data, train_test_split
from binaural_cues import get_ITD
import const_value
import binaural_cues
from frange import *

def DOA_train_gmm_model(dataset_path, freq):
    """ train gmm model for DOA

    Args:
        dataset_path, freq (const_value.frequency): dataset_path, freqency. 

    Returns:
        (list of dict, np.array), (list of dict, np.array): 
        (train_data_dict, train_data, output_label), (validation_data_dict, validation_data, validation_output_label)
        train_data format: (ILD, ITD)
    """

    data_dict = []
    train_data = []
    train_label = []

    if False:
        for freq in const_value.frequency:
            a, b, c = get_train_data(dataset_path=dataset_path, freq=freq, data_count=10)
            data_dict.extend(a)
            train_data.extend(b)
            train_label.extend(c)

        data_dict = np.array(data_dict)
        train_data = np.array(train_data)
        train_label = np.array(train_label)

    if True:
        data_dict, train_data, train_label = get_train_data(dataset_path=dataset_path, freq=freq, data_count=300)

    # print(data_dict)
    # print(train_data.shape, true_label.shape)
    # print(train_data[:5], true_label[:5])

    (train_data_dict, train_x, train_label), (validation_data_dict, validation_x, validation_label) \
        = train_test_split(data_dict, train_data, train_label)
    print(train_x.shape, train_label.shape, validation_x.shape, validation_label.shape)

    gmm = GaussianMixture(n_components=180 // 5 + 1, covariance_type='tied').fit(train_x)

    output_label = gmm.predict(train_x)
    result = gmm.predict_proba(train_x)

    print(result.shape)

    print('(true label, predicted label):', train_label[:10], output_label[:10], sep='\n')

    plt.figure(1)
    plt.scatter(train_x[:, 0], train_x[:, 1], c=train_label/15, s=30, cmap='viridis')

    plt.figure(2)
    plt.scatter(train_x[:, 0], train_x[:, 1], c=output_label/15, s=30, cmap='viridis')

    print('training score', gmm.score(train_x))
    print('validation score', gmm.score(validation_x))

    validation_output_label = gmm.predict(validation_x)
    # print(validation_output_label.shape)

    # plt.figure(3)
    # plt.scatter(validation_x[:, 0], validation_x[:, 1], c=validation_label/15, s=30, cmap='viridis')

    # plt.figure(4)
    # plt.scatter(validation_x[:, 0], validation_x[:, 1], c=validation_output_label/15, s=30, cmap='viridis')

    plt.show()

    print(DOA_gmm_evaluate(train_label, output_label))

    return (train_data_dict, train_x, train_label), (validation_data_dict, validation_x, validation_label)

def DOA_gmm_evaluate(true_label, output_label):
    """ 

    Args:
        true_label (_type_): _description_
        output_label (_type_): _description_
    
    Returns:
        _type_: _description_
    """

    label_count = np.zeros((37, 37))
    dic = {}

    print(true_label[:5], output_label[:5])

    data_count = len(true_label)
    # print('data_count: ', data_count)
    for i in range(data_count):
        label_count[true_label[i] // 5][output_label[i]] += 1
    for i in range(37):
        dic[np.argmax(label_count[i])] = i
        # print(label_count[i])

    print(dic)

    return "Accuracy: " + str(sum([x // 5 == dic[y] for x, y in zip(true_label, output_label)]) / len(true_label))

def DOA_build_CNN_model():

    input_shape = (2, )
    output_shape = 37

    model1 = Sequential([
        keras.layers.InputLayer(input_shape=input_shape),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(output_shape, activation='softmax')
    ])

    model1.summary()

    model1.compile(optimizer='adam', 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
                    metrics=['accuracy'])

    return model1

def DOA_train_CNN_model(data_dict, train_data, train_label, model):
    """_summary_

    Args:
        freq (_type_): _description_
    
    Returns:
        _type_: _description_
    """

    EPOCHS = 10


    # (train_data_dict, train_x, train_label), (validation_data_dict, validation_x, validation_label) \
    #     = train_test_split(data_dict, train_data, train_label)

    train_data_dict, train_x, train_label = data_dict, train_data, train_label


    history = model.fit(train_x, train_label // 5, epochs=EPOCHS, batch_size=None)
    results = model.evaluate(train_x, train_label // 5, verbose=2)
    # results = model.evaluate(validation_x, validation_label // 15, verbose=2)
    print(results)

    plt.figure(1)
    plt.scatter(train_x[:, 0], train_x[:, 1], c=train_label // 5, s=30)
    plt.show()

    return model

def get_data_mix_dataset(dataset_path, freq, data_count=300):

    data_dict = []
    train_data = []
    train_label = []
    for freq in const_value.frequency:
        a, b, c = get_train_data(dataset_path=dataset_path, freq=freq, data_count=data_count)
        data_dict.extend(a)
        train_data.extend(b)
        train_label.extend(c)

    data_dict = np.array(data_dict)
    train_data = np.array(train_data)
    train_label = np.array(train_label)

    return data_dict, train_data, train_label

    
def coordinate_train_coordinate_model(data_dict, train_data, train_label):
    """ user regression model to predict "r"

    model summary:
        input: (rms_energy)
        label: (r)

    Args:
        freq (_type_): _description_
    
    """

    model = DOA_build_CNN_model()
    model = DOA_train_CNN_model(data_dict, train_data, train_label, model=model)

    train_data = []
    for i in frange(len(data_dict)):
        train_data.append([data_dict[i]['rms_energy'], data_dict[i]['ITD'], data_dict[i]['ILD']])
    train_data = np.array(train_data)

    train_label = []
    for i in frange(len(data_dict)):
        train_label.append(data_dict[i]['r']) 
    train_label = np.array(train_label)

    regr = linear_model.Lasso()
    regr.fit(train_data, train_label)
    print('score', regr.score(train_data, train_label))


    # plt.figure(2)
    # plt.scatter(train_data, train_label, c=train_label/5, s=30, cmap='viridis')

    # plt.show()

    for data, label in zip(train_data[:50], train_label[:50]):
        print('(true, predict):', label, regr.predict([data]))

    return model, regr

def train_model(dataset_path=const_value.dataset_path[1], freq=const_value.frequency[4]):

    if True:
        data_dict, train_data, train_label = get_train_data(dataset_path=dataset_path, freq=freq, data_count=1500)
    if False:
        data_dict, train_data, train_label = get_data_mix_dataset(dataset_path=dataset_path, freq=freq, data_count=300)

    print(len(data_dict))

    (train_data_dict, train_x, train_label), (validation_data_dict, validation_x, validation_label) \
        = train_test_split(data_dict, train_data, train_label)


    model, regr = coordinate_train_coordinate_model(train_data_dict, train_x, train_label)

    print('in predict coordinate', '-' * 30)
    results = model.evaluate(validation_x, validation_label // 5, verbose=2)
    print('in predict coordinate', '-' * 30)

    print(validation_x[0], validation_data_dict[0])

    print(len(validation_x), len(validation_data_dict))


    errlist = []
    SD = 0

    for data, ddic in zip(validation_x, validation_data_dict):
        # print(data)
        theta = np.argmax(model.predict(np.array([data]))[0]) * 5
        r = regr.predict([[ddic['rms_energy'], ddic['ITD'], ddic['ILD']]])
        # print(f'r: {r} theta: {theta}')
        # print(f'true r: {ddic["r"]} theta: {ddic["theta"]}')
        predict_x = r * math.cos(theta * math.pi / 180)
        predict_y = r * math.sin(theta * math.pi / 180)
        x = ddic['x']
        y = ddic['y']

        print(f'(x, y): ({predict_x}, {predict_y})')
        print(f'true (x, y): ({x}, {y})')

        errlist.append(math.sqrt((predict_x - x) ** 2 + (predict_y - y) ** 2))

    SD /= len(validation_x) * (len(validation_x) - 1)
    SD = math.sqrt(SD)

    print(errlist)

    print(SD)

    return model, regr



if __name__ == '__main__':
    _, _ \
        = DOA_train_gmm_model(dataset_path=const_value.dataset_path[1], freq=const_value.frequency[3])

    # model = DOA_build_CNN_model()
    # (train_data, train_label), (validation_data, validation_label), model = DOA_train_CNN_model(dataset_path=const_value.dataset_path[1], freq=const_value.frequency[4], model=model)

    # coordinate_train_coordinate_model(freq=const_value.frequency[4])

    # model, regr = train_model(dataset_path=const_value.dataset_path[1], freq=const_value.frequency[3])

    # data = const_value.dataset_path[1]
    # x, y = predict_coordinate(data, model, regr)


