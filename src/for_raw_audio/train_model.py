import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Hide warning from tf

from frange import frange
#"""
# tensorflow is loading so slow QQ
import tensorflow as tf
import keras
from keras.models import Sequential
from keras import layers
print('Tensorflow loading finish ~ ٩(ˊᗜˋ*)و')
#"""
from time import sleep
import pandas as pd
import numpy as np
from PIL import Image

data_shape = (49152, 2)

def get_train_data(data_count=1000):
    """
    Read the csv file.
    data_count shouldn't be too large, since the memory won't fit. (All testcase requires 11.4 GB)
    maybe I should take random data_count data from the whole dataset?
    """
    file_path = '../../data/'
    df = pd.read_csv(file_path + 'dataset.csv')
    y = df['block'].to_numpy()

    y = y[:data_count]
    x = np.load('audio_sample.npy')
    x = x[:data_count]
    print(x.shape, y.shape)

    return (x, y)


def train_test_split(train_data, train_labels, test_split_ratio=0.2):
    """
    Split data into train and test.
    The dataset is already random, I don't think that random is needed.
    """

    test_split_count = int(len(train_data) * test_split_ratio)
    new_train_data = train_data[:-test_split_count]
    new_train_labels = train_labels[:-test_split_count]
    test_data = train_data[-test_split_count:]
    test_labels = train_labels[-test_split_count:]
    return (new_train_data, new_train_labels), (test_data, test_labels)

def build_model():

    input_shape = data_shape # wav length

    output = 16
    model1 = Sequential([
        keras.layers.Conv1D(32, 3, activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling1D(2),
        keras.layers.Dropout(0.2),
        keras.layers.Conv1D(32, 3, activation='relu'),
        keras.layers.MaxPooling1D(2),
        keras.layers.Dropout(0.2),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='tanh'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(output, activation='softmax')
    ])

    model1.summary()

    model1.compile(optimizer='adam', 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
                    metrics=['accuracy'])

    return model1

def train_model(model):
    """
    Training da model
    """
    
    EPOCHS = 10
    batch_size = None

    # early_stop = keras.callbacks.EarlyStopping(monitor='mean_absolute_error', patience=20)

    train_data, train_labels = get_train_data()
    print(train_data.shape, train_labels.shape)

    (new_train_data, new_train_labels), (test_data, test_labels) = train_test_split(train_data, train_labels)
    print(new_train_data.shape, new_train_labels.shape, test_data.shape, test_labels.shape)

    history = model.fit(new_train_data, new_train_labels, epochs=EPOCHS, batch_size=batch_size)

    print(test_data[0].shape, train_data[0].shape)

    results = model.evaluate(test_data, test_labels, batch_size=None)
    print(results)

    # prediction = model.predict(test_data[0])

    return (model, history)

def evaluate(model):
    model.predict()
    pass

def plot_history(history):

    history_dict=history.history
    loss_values=history_dict['loss']
    acc_values=history_dict['accuracy']
    val_loss_values = history_dict['val_loss']
    val_acc_values=history_dict['val_accuracy']
    epochs=frange(1,11)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
    ax1.plot(epochs,loss_values,'bo',label='Training Loss')
    ax1.plot(epochs,val_loss_values,'orange', label='Validation Loss')
    ax1.set_title('Training and validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2.plot(epochs,acc_values,'bo', label='Training accuracy')
    ax2.plot(epochs,val_acc_values,'orange',label='Validation accuracy')
    ax2.set_title('Training and validation accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.show()


